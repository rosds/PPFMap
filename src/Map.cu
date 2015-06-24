#include <PPFMap/Map.h>
#include <PPFMap/PPFEstimationKernel.h>


struct extract_hash_key : public thrust::unary_function<uint64_t, uint32_t> {
    __host__ __device__
    uint32_t operator()(const uint64_t ppf_code) const {
        return static_cast<uint32_t>(ppf_code >> 32);
    }
};


struct copy_element_by_index : public thrust::unary_function<uint32_t, uint32_t> {
    const uint32_t* ppf_index_ptr;

    copy_element_by_index(thrust::device_vector<uint32_t> const& vec) 
        : ppf_index_ptr(thrust::raw_pointer_cast(vec.data())) {}

    __host__ __device__
    uint32_t operator()(const uint32_t index) const {
        return ppf_index_ptr[index];
    }
};


struct VotesExtraction {
    const float discretization_angle;

    const float* alpha_s;

    const uint64_t* ppf_codes;
    const bool* ppf_found;
    const uint32_t* ppf_index;
    const uint32_t* ppf_count;
    const uint32_t* insert;

    uint32_t* votes_ptr;

    VotesExtraction(const thrust::device_vector<float>& alphas,
                    const thrust::device_vector<uint64_t>& map_codes,
                    const thrust::device_vector<bool>& map_found,
                    const thrust::device_vector<uint32_t>& map_index,
                    const thrust::device_vector<uint32_t>& map_count,
                    const thrust::device_vector<uint32_t>& insert_votes,
                    const float disc_angle,
                    thrust::device_vector<uint32_t>& votes)
        : alpha_s(thrust::raw_pointer_cast(alphas.data()))
        , ppf_codes(thrust::raw_pointer_cast(map_codes.data()))
        , ppf_found(thrust::raw_pointer_cast(map_found.data()))
        , ppf_index(thrust::raw_pointer_cast(map_index.data()))
        , ppf_count(thrust::raw_pointer_cast(map_count.data()))
        , insert(thrust::raw_pointer_cast(insert_votes.data()))
        , discretization_angle(disc_angle)
        , votes_ptr(thrust::raw_pointer_cast(votes.data())) {}

    __device__
    void operator()(const int i) {
        if (ppf_found[i]) {
            for (int vote_idx = 0; vote_idx < ppf_count[i]; vote_idx++) {

                uint64_t model_ppf_code = ppf_codes[ppf_index[i] + vote_idx];

                uint16_t model_index = static_cast<uint16_t>(model_ppf_code >> 16 & 0xFFFF);
                float alpha_m = static_cast<float>(model_ppf_code & 0xFFFF) * discretization_angle;

                uint16_t alpha = static_cast<uint16_t>((alpha_m - alpha_s[i]) / discretization_angle);

                uint32_t vote = static_cast<uint32_t>(model_index) << 16 |
                                static_cast<uint32_t>(alpha);

                votes_ptr[insert[i] + vote_idx] =  vote;
            }
        }
    }
};


struct PPFMapSearch {
    const std::size_t n;
    const uint32_t* hash_list;

    const uint32_t* hash_keys;
    const uint32_t* ppf_index;
    const uint32_t* ppf_count;

    bool* out_found;
    uint32_t* out_index;
    uint32_t* out_count;

    PPFMapSearch(const thrust::device_vector<uint32_t>& hl,
                 const thrust::device_vector<uint32_t>& map_hash_keys,
                 const thrust::device_vector<uint32_t>& map_ppf_index,
                 const thrust::device_vector<uint32_t>& map_ppf_count,
                 thrust::device_vector<bool>& result_found,
                 thrust::device_vector<uint32_t>& result_index,
                 thrust::device_vector<uint32_t>& result_count)
        : n(map_hash_keys.size())
        , hash_list(thrust::raw_pointer_cast(hl.data()))
        , hash_keys(thrust::raw_pointer_cast(map_hash_keys.data()))
        , ppf_index(thrust::raw_pointer_cast(map_ppf_index.data()))
        , ppf_count(thrust::raw_pointer_cast(map_ppf_count.data()))
        , out_found(thrust::raw_pointer_cast(result_found.data()))
        , out_index(thrust::raw_pointer_cast(result_index.data()))
        , out_count(thrust::raw_pointer_cast(result_count.data())) {}

    __device__
    void operator()(const int i) {
        const uint32_t hk = hash_list[i]; 

        out_found[i] = false;
        out_index[i] = 0;
        out_count[i] = 0;
    
        int l = 0;
        int r = n;
        int m = (l + r) / 2;

        while (l < r) {
            if (hk < hash_keys[m]) {
                r = m;
            }
            else if (hk > hash_keys[m]) {
                l = m + 1;
            }
            else {
                out_found[i] = true;
                out_index[i] = ppf_index[m];
                out_count[i] = ppf_count[m];
                break; 
            } 
            m = (l + r) / 2; 
        }
    }
};


/** \brief Computes the PPF features for the input cloud.
 *  \param[in] h_points Host vector with the 3D information of the points.
 *  \param[in] h_normals Host vector with the normals of each point.
 *  \param[in] disc_dist Discretization factor for pair distance.
 *  \param[in] disc_angle Discretization factor for angles.
 */
ppfmap::Map::Map(const pcl::cuda::Host<float3>::type& h_points,
                 const pcl::cuda::Host<float3>::type& h_normals,
                 const float disc_dist,
                 const float disc_angle)
    : discretization_distance(disc_dist)
    , discretization_angle(disc_angle) {

    const std::size_t number_of_points = h_points.size();
    const std::size_t number_of_pairs = number_of_points * number_of_points;

    float affine[12];

    pcl::cuda::Device<float3>::type d_points(h_points);
    pcl::cuda::Device<float3>::type d_normals(h_normals);

    ppf_codes.resize(number_of_pairs);

    float max_distance = 0.0f;
    for (int i = 0; i < number_of_points; i++) {
        const float3 point_position = h_points[i];
        const float3 point_normal = h_normals[i];

        ppfmap::getAlignmentToX(point_position, point_normal, &affine);

        ppfmap::PPFEstimationKernel ppfe(point_position, point_normal, i,
                                         discretization_distance,
                                         discretization_angle,
                                         affine);

        thrust::transform(d_points.begin(), d_points.end(),
                          d_normals.begin(),
                          ppf_codes.begin() + i * number_of_points,
                          ppfe);

        float max_pair_dist = ppfmap::maxDistanceToPoint<pcl::cuda::Device>(point_position, d_points);

        if (max_distance < max_pair_dist) {
            max_distance = max_pair_dist; 
        }
    }
    
    cloud_diameter = max_distance;

    thrust::sort(ppf_codes.begin(), ppf_codes.end());

    thrust::device_vector<uint32_t> hash_tmp(number_of_pairs);

    // copy the hash keys to a separate vector
    thrust::transform(
        ppf_codes.begin(), 
        ppf_codes.end(), 
        hash_tmp.begin(),
        extract_hash_key()
    );

    hash_keys.resize(number_of_pairs);
    ppf_count.resize(number_of_pairs);

    thrust::pair<thrust::device_vector<uint32_t>::iterator, 
                 thrust::device_vector<uint32_t>::iterator> end;

    // Count the number of similar keys
    end = thrust::reduce_by_key(hash_tmp.begin(), hash_tmp.end(),
                                thrust::make_constant_iterator(1),
                                hash_keys.begin(),
                                ppf_count.begin());

    const size_t unique_hash_keys = end.first - hash_keys.begin();

    // Fix the vectors to size
    hash_keys.resize(unique_hash_keys);
    hash_keys.shrink_to_fit();
    ppf_count.resize(unique_hash_keys);
    ppf_count.shrink_to_fit();

    ppf_index.resize(unique_hash_keys);

    // Set the array with the indices to the first instance of each key in the 
    // codes array.
    thrust::exclusive_scan(ppf_count.begin(), ppf_count.end(),
                           ppf_index.begin());

    max_votes = thrust::reduce(ppf_count.begin(), ppf_count.end(), 
                               0, thrust::maximum<uint32_t>());
}


/** \brief Performs the voting and accumulation for the ppf list provided 
 * and returns the best point index and resulting alpha.
 *  \param[in] hash_list List of hashed ppf features to query
 *  \param[in] alpha_s Angle to align the reference point to the x axis.
 *  \param[out] m_idx Best matching index in Hough voting space.
 *  \param[out] alpha Resulting angle after combining the alpha_s and 
 *  alpha_m.
 *  \param[out] max_votes The number of pairs supporting the m_idx and alpha 
 *  parameters.
 */
void ppfmap::Map::searchBestMatch(const thrust::host_vector<uint32_t> hash_list, 
                                  const thrust::host_vector<float> alpha_s_list,
                                  int& m_idx, float& alpha, int& max_votes) {


    thrust::device_vector<uint32_t> d_hash_list(hash_list);
    thrust::device_vector<float> d_alpha_s_list(alpha_s_list);

    thrust::device_vector<bool> d_key_found(d_hash_list.size());
    thrust::device_vector<uint32_t> d_ppf_index(d_hash_list.size());
    thrust::device_vector<uint32_t> d_ppf_count(d_hash_list.size());
    thrust::device_vector<uint32_t> d_insert_pos(d_hash_list.size());

    PPFMapSearch m_search(d_hash_list, 
                          hash_keys, ppf_index, ppf_count, 
                          d_key_found, d_ppf_index, d_ppf_count);

    thrust::counting_iterator<int> it(0);
    thrust::for_each(it, it + hash_list.size(), m_search);

    uint64_t votes_total = thrust::reduce(d_ppf_count.begin(), d_ppf_count.end(), 
                                          0, thrust::plus<uint32_t>());

    // This sets the position where to start inserting the votes of each ppf
    thrust::exclusive_scan(d_ppf_count.begin(), d_ppf_count.end(), d_insert_pos.begin());

    thrust::device_vector<uint32_t> votes(votes_total);
    thrust::device_vector<uint32_t> unique_votes(votes_total);
    thrust::device_vector<uint32_t> vote_count(votes_total);

    VotesExtraction write_votes(d_alpha_s_list, 
                                ppf_codes, 
                                d_key_found, d_ppf_index, 
                                d_ppf_count, d_insert_pos,
                                discretization_angle,
                                votes);

    thrust::for_each(it, it + hash_list.size(), write_votes);

    thrust::sort(votes.begin(), votes.end());

    thrust::pair<thrust::device_vector<uint32_t>::iterator, 
                 thrust::device_vector<uint32_t>::iterator> end;

    end = thrust::reduce_by_key(votes.begin(), votes.end(), 
                                thrust::make_constant_iterator(1), 
                                unique_votes.begin(), 
                                vote_count.begin());

    unique_votes.resize(end.first - unique_votes.begin());
    vote_count.resize(end.second - vote_count.begin());

    thrust::device_vector<uint32_t>::iterator iter =
          thrust::max_element(vote_count.begin(), vote_count.end());

    int position = iter - vote_count.begin();

    const uint32_t winner = unique_votes[position];

    m_idx = static_cast<int>(winner >> 16);
    alpha = static_cast<float>(winner & 0xFFFF) * discretization_angle;
    max_votes = static_cast<int>(*iter);
}

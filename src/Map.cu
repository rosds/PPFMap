#include <PPFMap/Map.h>
#include <PPFMap/PPFEstimationKernel.h>


struct extract_hash_key : thrust::unary_function<uint64_t, uint32_t> {
    __host__ __device__
    uint32_t operator()(const uint64_t ppf_code) const {
        return static_cast<uint32_t>(ppf_code >> 32);
    }
};


struct copy_element_by_index : thrust::unary_function<uint32_t, uint32_t> {
    const uint32_t* ppf_index_ptr;

    copy_element_by_index(thrust::device_vector<uint32_t> const& vec) 
        : ppf_index_ptr(thrust::raw_pointer_cast(vec.data())) {}

    __host__ __device__
    uint32_t operator()(const uint32_t index) const {
        return ppf_index_ptr[index];
    }
};


struct write_votes {
    const uint64_t* model_ppf_ptr;
    const float discretization_angle;
    uint32_t* votes_ptr;

    write_votes(thrust::device_vector<uint64_t> const& model_ppf,
                const float disc_angle,
                thrust::device_vector<uint32_t> &votes)
        : model_ppf_ptr(thrust::raw_pointer_cast(model_ppf.data()))
        , discretization_angle(disc_angle)
        , votes_ptr(thrust::raw_pointer_cast(votes.data())) {}

    template <class Tuple> __device__
    void operator()(Tuple t) {

        const uint32_t insert_position = thrust::get<0>(t); 
        const bool     key_found = thrust::get<1>(t); 
        const uint32_t ppf_index = thrust::get<2>(t); 
        const uint32_t ppf_count = thrust::get<3>(t); 
        const float alpha_s = thrust::get<4>(t); 

        if (key_found) {
            for (int vote_idx = 0; vote_idx < ppf_count; vote_idx++) {

                uint64_t model_ppf_code = model_ppf_ptr[ppf_index + vote_idx];

                uint16_t model_index = static_cast<uint16_t>(model_ppf_code >> 16 & 0xFFFF);
                float alpha_m = static_cast<float>(model_ppf_code & 0xFFFF) * discretization_angle;

                uint16_t alpha = static_cast<uint16_t>((alpha_m - alpha_s) / discretization_angle);

                uint32_t vote = static_cast<uint32_t>(model_index) << 16 |
                                static_cast<uint32_t>(alpha);

                votes_ptr[insert_position + vote_idx] =  vote;
            }
        }
    }
};


ppfmap::Map::Map(const pcl::cuda::PointCloudSOA<pcl::cuda::Host>::Ptr cloud,
                 const pcl::cuda::PointCloudSOA<pcl::cuda::Host>::Ptr normals,
                 const float disc_dist,
                 const float disc_angle)
    : discretization_distance(disc_dist)
    , discretization_angle(disc_angle) {

    const size_t number_of_points = cloud->size();
    const size_t number_of_pairs = number_of_points * number_of_points;

    float affine[12];

    pcl::cuda::PointCloudSOA<pcl::cuda::Device> d_cloud;
    pcl::cuda::PointCloudSOA<pcl::cuda::Device> d_normals;

    d_cloud << *cloud;
    d_normals << *normals;

    ppf_codes.resize(number_of_pairs);

    for (int i = 0; i < number_of_points; i++) {
    
        const float3 point_position = make_float3(cloud->points_x[i],
                                                  cloud->points_y[i],
                                                  cloud->points_z[i]);

        const float3 point_normal = make_float3(normals->points_x[i],
                                                normals->points_y[i],
                                                normals->points_z[i]);

        ppfmap::getAlignmentToX(point_position, point_normal, &affine);

        ppfmap::PPFEstimationKernel<pcl::cuda::Device> 
            ppfe(point_position, point_normal, i,
                 discretization_distance,
                 discretization_angle,
                 affine);

        thrust::transform(d_cloud.zip_begin(), d_cloud.zip_end(),
                          d_normals.zip_begin(),
                          ppf_codes.begin() + i * cloud->size(),
                          ppfe);
    }

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


void ppfmap::Map::searchBestMatch(const thrust::host_vector<uint32_t> hash_list, 
                                  const thrust::host_vector<float> alpha_s_list,
                                  int& m_idx, float& alpha) {


    thrust::device_vector<uint32_t> d_hash_list = hash_list;
    thrust::device_vector<float> d_alpha_s_list = alpha_s_list;

    thrust::device_vector<bool> d_key_found(d_hash_list.size());
    thrust::device_vector<uint32_t> d_key_index(d_hash_list.size());
    thrust::device_vector<uint32_t> d_ppf_index(d_hash_list.size());
    thrust::device_vector<uint32_t> d_ppf_count(d_hash_list.size());
    thrust::device_vector<uint32_t> d_insert_pos(d_hash_list.size());

    thrust::binary_search(hash_keys.begin(), hash_keys.end(),
                          d_hash_list.begin(), d_hash_list.end(),
                          d_key_found.begin());

    thrust::lower_bound(hash_keys.begin(), hash_keys.end(),
                        d_hash_list.begin(), d_hash_list.end(),
                        d_key_index.begin());

    thrust::transform(d_key_index.begin(), d_key_index.end(), 
                      d_ppf_index.begin(), 
                      copy_element_by_index(ppf_index));

    thrust::transform(d_key_index.begin(), d_key_index.end(), d_ppf_count.begin(), 
                      copy_element_by_index(ppf_count));

    uint64_t votes_total = thrust::reduce(d_ppf_count.begin(), d_ppf_count.end(), 
                                          0, thrust::plus<uint64_t>());

    // This sets the position where to start inserting the votes of each ppf
    thrust::exclusive_scan(d_ppf_count.begin(), d_ppf_count.end(), d_insert_pos.begin());

    thrust::device_vector<uint32_t> votes(votes_total);
    thrust::device_vector<uint32_t> unique_votes(votes_total);
    thrust::device_vector<uint32_t> vote_count(votes_total);

    thrust::for_each(
        thrust::make_zip_iterator(
            thrust::make_tuple(
                d_insert_pos.begin(), 
                d_key_found.begin(),
                d_ppf_index.begin(),
                d_ppf_count.begin(),
                d_alpha_s_list.begin()
            )
        ),          
        thrust::make_zip_iterator(
            thrust::make_tuple(
                d_insert_pos.end(), 
                d_key_found.begin(),
                d_ppf_index.end(),
                d_ppf_count.end(),
                d_alpha_s_list.end()
            )
        ),          
        write_votes(ppf_codes, discretization_angle, votes)
    );

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

    m_idx = static_cast<int>(unique_votes[position] >> 16);
    alpha = static_cast<float>(unique_votes[position] & 0xFFFF) * discretization_angle;
}

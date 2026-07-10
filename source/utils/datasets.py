from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner, NaturalIdPartitioner

def get_partitioner(distribution, num_partitions, alpha, partition_by = None, femnist=False):
    """Returns either a IidPartitioner for IID distribution or DirichletPartitioner for Non-IID distribution"""

    if str.lower(distribution) == 'iid':
        return IidPartitioner(num_partitions=num_partitions)
    
    elif str.lower(distribution) == 'non-iid':
        if femnist:
            return NaturalIdPartitioner(
                partition_by='writer_id'
            )
        else:
            return DirichletPartitioner(
                num_partitions=num_partitions,
                partition_by=partition_by,
                alpha=alpha,
                seed=42
            )
    
    else:
        return ValueError(f"Invalid distribution: {distribution}. Please select 'iid' or 'non-iid'.")
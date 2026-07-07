from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner

def get_partitioner(distribution, num_partitions, alpha):
    """Returns either a IidPartitioner for IID distribution or DirichletPartitioner for Non-IID distribution"""

    if str.lower(distribution) == 'iid':
        return IidPartitioner(num_partitions=num_partitions)
    
    elif str.lower(distribution) == 'non-iid':
        return DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by='label',
            alpha=alpha,
            seed=42
        )
    
    else:
        return ValueError(f"Invalid distribution: {distribution}. Please select 'iid' or 'non-iid'.")

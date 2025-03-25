from typing import List, Union
import anndata


def get_cko_data(data: anndata.AnnData, cko_gene_names: Union[List[str], str]) -> anndata.AnnData:
    """
    Get the adata with the cko genes.
    """
    cko_gene_names = [cko_gene_names.lower()] if isinstance(cko_gene_names, str) else [gene.lower() for gene in cko_gene_names]
    data.var_names = data.var_names.str.lower()
    gene_index = [data.var_names.get_loc(gene_name) for gene_name in cko_gene_names]
    data.X[:, gene_index] = 0.
    print(f"CKO genes {cko_gene_names} are set")
    
    return data

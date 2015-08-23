# NTriPath

NTriPath is a method to integrate somatic mutations with biological prior knowledge (e.g., protein-protein interaction networks, pathway database) to detect cancer-type specific altered pathways by somatic mutations across cancers.
The followings are details of NTriPath workflow:
Four types of data were used as input for our algorithm. First, we generated a binary matrix X of patients x genes, with ‘1’ indicating a mutation and ‘0’ no mutation. Second, we constructed gene-gene interaction networks A. Third, we incorporated a pathway database V_0 (e.g., conserved 4,620 subnetworks across species12). Fourth, we included clinical data on the patient's tumor type U. NTriPath produces two matrices as output; 1) altered pathways by mutated genes V and 2) altered pathways by cancer type matrix S. The use of both large-scale somatic mutation profiles and gene-gene interaction networks enabled NTriPath to identify cancer-related pathways containing known cancer genes mutated at different frequencies across cancers with newly added member genes according to high network connectivity. Finally we use the altered pathways by cancer type matrix S to identify altered pathways that are specific for each cancer type.

If you have any questions, please send an email via compbio.utsw@gmail.com.

===========

Simple steps to run this Matlab code
-------
- Open the Matlab 
- Check if the main file and data directory are in your working directory 
- Set the path in Matlab
- Type main in the command window or click the run button in the editor tap 

Reference 
-------
- [Computational Network Biology Lab @UTSW](http://www.taehyunlab.org/#!ntripath/c8c5)

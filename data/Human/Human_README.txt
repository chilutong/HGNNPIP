The description of the data files associated with Human was described as follows.
The details can be also referred from the link: https://github.com/chilutong/HGNNPIP/blob/master/data
------------------------------------------------------------------------------------------------------------------------------------------------
1) Human_uniprot:     There are there columns in this file, including (RowId), ProteinID, UniprotID.

2) Human_sequence:    Each line represents the amino acid sequence of a protein.

3) Orig_network.      This file is the raw PPI dataset, which is directly collected from Pan's work. (10364 proteins; 36630 positive interactions)
                      There are four columns in this file, including (RowId), ProteinID, ProteinID, Interacting_state.                       
                      The possible values for Interacting_state: 1 for 'interaction'; 0 for 'no-interaction'. 

4) Adj_network.       This file is the adjusted PPI dataset with negative sample generation via RanNS.
                      There are four columns in this file, including (RowId), ProteinID, ProteinID, Interacting_state.                       
                      The possible values for Interacting_state: 1 for 'interaction'; 0 for 'no-interaction'. 

5) Group 1:           Also named as ’NTS1’ in our paper.
                      There are four columns in this file, including (RowId), UniprotID, UniprotID, '0'.
                      ‘0’ means this is a negative instance (no-interaction).

6) Group 2:           Also named as ‘NTS2’ in our paper.
                      There are four columns in this file, including (RowId), UniprotID, UniprotID, '0'.
                      ‘0’ means this is a negative instance (no-interaction).


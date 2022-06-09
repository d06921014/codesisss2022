# Submission ID 35 for CODES+ISSS22
Enhancing Reliability and Security: A Configurable Poisoning PUF against Modeling Attacks

###Package requirements
- Please check the file "requirements.txt"

###Data Preparation

- Default path for challenge-response pairs is "/dataset"
- Run "python dataPreparation.py" to initialize randomly generated challenges.
- Generate responses for the corresponding PUFs. For instance "python getAPUFResponse.py" for APUF's responses

###File Descriptions:
|   Files| Descriptions  |
| ------------ | ------------ |
|  apuf_lib.py |PUF library. Implementations for APUF, XOR APUF, CP PUF, IPUF, DCH PUF.|
|  quality_metrics.py |  Measure PUF's uniformity and uniqueness. |
|  getPUFResponse.py|  Initiate the PUF instances and obtain responses by challenges.|
| dnn_puf.py |Modeling attack using deep learning model.|
|FR_FA_theoretic_result.py|Theoretical analysis for false rejection rate (FRR) and false acceptance rate (FAR) of the proposed CP PUF protocol.|
|authentication_test.py|Simulate the authentication for the CP PUF's protocol under different noise level. Reliability of CP PUF and two internal APUFs, FRR, FAR are reported.|
|reliability_test.py|Evaluate reliability of PUF instances under different noise level.|
|InvertFunc.py|Implementation of inversion functions for the AAPUF.|
|genSelectedData.py|Generate baised challenges to evaluate the ML resistance of AAPUF under chosen challenge attack.|
|sac_test.py|Evalutate the SAC property of a PUF instance.|
|utils.py|Tools for challenge transformations and theoretical analysis|
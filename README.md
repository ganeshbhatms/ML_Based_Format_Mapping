# ML_Based_Format_Mapping
Machine Learning based nvoice format matching module exposed as a REST Service. It takes an input and a target out fields list and tries to map the fields based on field name similarity. The matching algorithms need to consider basic String similarity, Acronyms, short forms and so on.

The API will have two modes of operation
1. Supervised mode
  * API takes a set of source and target fields and returns the mapping with matching confidence score
  * The service learns the mapping.  
   	We iterate this process with 3-4 source format fields for the same target format. 
2. Unsupervised mode
  * We give a new source format fields along with same target format fields
  * The service should predict the mapping for source fields
  
### simtext
This file contains `CosineSimilarty` funtion for finding similarity between two text and `FitModel` for KNN algorithm

### REST APIs
supervised and unsupervised files are format matching module for training KNN algorith exposed as a REST Service. Predict file is for prediction of new texts.

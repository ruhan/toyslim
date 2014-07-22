SLIM and SSLIM TOY implementations
----------

We have implemented SLIM [1]_ and SSLIM [2]_ (more specifically cSLIM) in a "toy" way, that's useful ONLY to study these methods, it is absolutely impossible to use these implementations on production, because of performance and some static aspects put on the code. 

The code implemented here is also based on mrec (https://github.com/Mendeley/mrec). We preferred to implement a toy version because we find some difficulties to debug mrec code because its complexities related to be a production tool. If you want to use it in production, we really suggest you to go to this excelent implementation of SLIM done by the Mendeley Team.


We've also implemented some new ideas to extend these methods in some specific cases.


References
----------
.. [1] Xia Ning and George Karypis (2011). SLIM: Sparse Linear Methods for Top-N Recommender Systems. ICDM, 2011.
.. [2] Xia Ning and George Karypis (2012). Sparse linear methods with side information for top-n recommendations. RecSys 2012.

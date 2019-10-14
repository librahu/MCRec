# MCRec
Source code for KDD 2018 paper "Leverage Meta-path based Context for Top-N Recommendation with a Neural Co-Attention Model"

# Requirements

* numpy

* scipy

* Tensorflow-gpu (1.2.1) or Theano (1.0.1)

* Keras (2.1.1)

* My machine with two GPUs (NVIDIA GTX-1080 *2) and two CPUs (Intel Xeon E5-2690 * 2)

# Reference

@inproceedings{

> author = {Binbin Hu, Chuan Shi, Wayne Xin Zhao and Philip S. Yu.},

> title = {Leverage Meta-path based Context for Top-N Recommendation with a Neural Co-Attention Model},

> booktitle = {Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},

> year = {2018},

> url = {https://dl.acm.org/citation.cfm?id=3219965},

> publisher = {ACM},

> address = {London, United Kingdom},

> keywords = {Recommender System, Heterogeneous Information Network, Deep Learning, Attention Mechanism},

}

# More Details

**Dataset**

We provide one processed dataset : MovieLens 100k (ml-100k). For each user, we treat the latest interaction as the test set and remaining data for training. Besides, we randomly select 50 movies that are not interacted by the user and rank the test movie amongst the 50 movies.

train.rating:

- Train file.
- Each Line is a training instance: userID itemID rating timestamp (if have)

test.rating:

- Test file (positive instances).
- Each Line is a testing instance: userID rated_itemID_1 rated_itemID_2 rated_itemID_3...

test.negative

- Test file (negative instances).
- Each line corresponds to the line of test.rating, containing 50 negative samples.
- Each line is in the format: (userID,rated_itemID_1,rated_itemID_2,rated_itemID_3,...)\t negativeItemID1\t negativeItemID2 ...

**Metapath Construction**

The data we utilize is located in [HIN Datasets](https://github.com/librahu/HIN-Datasets-for-Recommendation-and-Network-Embedding/tree/master/Movielens) or you can obtain it by executing the dataProcessing.py.

```python
python dataProcessing.py
```

Run metapath construction : 

The metapathbasedPathSampleForMovielens.py constructs different meta-paths for the following model training.  

```python
python metapathbasedPathSampleForMovielens.py --walk_num 5 --metapath umtm
```

Here,  

- uafile :  user_age.dat , which extracts from original u.user, 

  	0~10 : 1;  11 ~20 : 2;  21~30 : 3; 31~40 : 4；41~50 : 5; 51~60 : 6

- uofile :  user_occupation.dat, which is the relation between u.user and u.occupation

- mtfile :  movie_genre.dat, which the relation between u.user and u.genre 

- uufile :  user_user(knn).dat

- mmfile : movie_movie(knn).dat

Output：

- the meaningful meta-path samples，which constructs “umtm / ummm / umum / uuum”  meta-path samples. 
- the filename's format is "dataset"."metapath"_"walk_length"\_"K"

**Example to run the codes.**

The parameter setting is set on line 476-484 of file MCRec, and the dataset for which is Movielens-100K.  Be sure to refer to it while experimenting with different datasets.

Run MCRec:

```python
python MCRec.py --dataset ml-100k --epochs 30 --batch_size 256 --learner adam --lr 0.001 --latent_dim 128 --latent_layer_dim [512, 256, 128, 64] --num_neg 4
```

**Remarkable** 

- the pretrain embedding will be useful for the model training and meta-path construction
- the negative sample is necessary for our model training, which is located in the line 413 of MCRec.py


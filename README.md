# bird-eye-view
This product uses ML technologies to compare the visual similarities (*geometrical, color, etc*) of images.
It is intended to be used on ecommerce sites as a product recommender system.
Models are trained with the VGG16 model which is pretrained on the imagenet dataset.

## To run
The program works as follows:

 - feture extraction is done in the `src/features/build_features.py` and saved into `data/embeddings/embeddings.npy`
 - the features are read in the `src/models/train_model.py`. The program uses Facebook's Similarity Search `faiss` to calculate the visual similarity of our candidate element
The api is build with flask and to run the program use: `python server.py`
The homepage `127.0.0.1:8080` returns a json response of all the products in our dataset'

### Predicting recommendations for a product
To get the recommendation for a product: get the `product-id` from the `id` column of our dataset.
Use it on the `127.0.0.1:8080/product/<product_id>` replace `product-id` with the value of `id` of the product on our dataframe.
The engine returns the 10 most similar products by default
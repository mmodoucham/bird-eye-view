import numpy as np
import faiss


df_embs = np.load(
    '/home/mmodoucham/bird-eye-view/data/embeddings/embeddings.npy')
d = df_embs.shape[1]
df_embs_np = df_embs.copy(order='C')


def predict(product_id, k=10):

    # #FAISS predictions with L2 distance
    # xq = df_embs_np[product_id].reshape(1,-1)
    # index = faiss.IndexFlatL2(d)
    # index.add(df_embs_np)

    # k = 4
    # D, I = index.search(xq, k)
    # print(I[0])

    #FAISS training with partitioned index
    xq = df_embs_np[product_id].reshape(1, -1)
    nlist = 100
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    index.train(df_embs_np)
    index.add(df_embs_np)
    index.nprobe = 10
    D, I = index.search(xq, k)
    return I[0]

import numpy as np


def normalize_db(db):
    subject, sample, row, col = db.shape
    db = db.astype(np.float)
    for i in range(subject):
        for j in range(sample):
            im = db[i, j]
            im -= im.min()
            im /= im.max()
    return db


def normalize_fvc(src, des):
    loaded = np.load(src)
    db_list = [loaded['DB{}'.format(i + 1)] for i in range(4)]

    for i in range(4):
        db_list[i] = normalize_db(db_list[i])

    np.savez_compressed(des,
                        DB1=db_list[0],
                        DB2=db_list[1],
                        DB3=db_list[2],
                        DB4=db_list[3],
                        )


if __name__ == '__main__':
    normalize_fvc('dataset/fvc_181_181.npz', 'dataset/fvc_normalize_181_181.npz')

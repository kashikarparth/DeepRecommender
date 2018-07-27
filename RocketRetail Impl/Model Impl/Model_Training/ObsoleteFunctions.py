# Obsolete Functions from the run.py file

def coo_submatrix_pull(matr, rows, cols):
    if type(matr) != sp.coo_matrix:
        raise TypeError('Matrix must be sparse COOrdinate format')
    gr = -1 * np.ones(matr.shape[0])
    gc = -1 * np.ones(matr.shape[1]) 
    lr = len(rows)
    lc = len(cols) 
    ar = np.arange(0, lr)
    ac = np.arange(0, lc)
    gr[rows[ar]] = ar
    gc[cols[ac]] = ac
    mrow = matr.row
    mcol = matr.col
    newelem = (gr[mrow] > -1) & (gc[mcol] > -1)
    newrows = mrow[newelem]
    newcols = mcol[newelem]
    return sp.coo_matrix((matr.data[newelem], np.array([gr[newrows],gc[newcols]])),(lr, lc))

def data_preprocessing(x):
    x = x.tocoo()
    for i in range(9):
        string = 'part'+ str(i) + '.npz'
        sp.save_npz(string,coo_submatrix_pull(x,np.arange(i*4000,(i+1)*4000),np.arange(50000)))
        print("part ",i," saved")


def output_checker():
    sparse1 = np.clip(np.round(model.predict(train_data)),1,5)                         
    print(sparse1)
    idx = (-np.round(sparse1[1230])).argsort()[:10]
    print(idx)

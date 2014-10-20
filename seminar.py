def blue_peter():
    import menpo.io as mio
    import h5it
    from menpo.visualize.image import glyph
    from menpo.feature import hog
    import matplotlib.pyplot as plt
    # Loading the pre-built HOG AAM
    import cPickle as pickle

    with open('/Users/pts08/hog_lfpw_aam.pkl', 'rb') as f:
        hog_aam = pickle.load(f)
    
    #hog_aam = h5it.load('/Users/pts08/sparse_hog.hdf5')
    print('Here is one I made earlier!')

    bp = mio.import_image('blue_peter.jpg')
    hog_blue_peter = hog(bp)

    plt.figure()

    plt.subplot(121)
    bp.view()
    plt.axis('off')
    plt.gcf().set_size_inches(11, 11)
    plt.title('RGB')

    plt.subplot(122)
    glyph(hog_blue_peter).view()
    plt.axis('off')
    plt.gcf().set_size_inches(11, 11)
    plt.title('HOG')

    return hog_aam

import cv2
import numpy as np
from imageclassifier import ImageClassifier



n_clusters = [3, 4, 5, 6, 7, 8]
kmeans_keys = [
    [0], [1], [2],
    [0, 1], [0, 2], [1, 2],
    [0, 1, 2] 
]

sorting_lambdas = [
    lambda pixel: pixel[0],
    lambda pixel: pixel[1],
    lambda pixel: pixel[2],
    lambda pixel: sum(pixel),
    lambda pixel: max(pixel)
]

cl_sorting_lambdas = [
    lambda cluster: cluster[0][0][0],
    lambda cluster: cluster[0][0][1],
    lambda cluster: cluster[0][0][2],
    lambda cluster: sum(cluster[0][0]),
    lambda cluster: max(cluster[0][0])
]


coeffs = []
for i in range(5):
    for j in range(5):
        for k in range(5):
            coeffs.append([i, j, k])



sorting_keys = [i for i in range(len(sorting_lambdas))]

colorspaces = [None, cv2.COLOR_BGR2HSV, cv2.COLOR_BGR2LAB, cv2.COLOR_BGR2HLS]

def str_colorspace(colorspace):
    if colorspace == None:
        return "BGR"
    if colorspace == cv2.COLOR_BGR2HSV:
        return "HSV"
    if colorspace == cv2.COLOR_BGR2LAB:
        return "LAB"
    if colorspace == cv2.COLOR_BGR2HLS:
        return "HLS"

def save(folder, img, n_cluster, key, color_in, sorting_key, color_sort):

    filename = folder + "/c{0}_k".format(n_cluster)
    filename = filename + '-'.join([str(s) for s in key])
    filename = filename + '_' + str_colorspace(color_in) + "_"
    filename = filename + 's{0}_'.format(sorting_key)
    filename = filename + str_colorspace(color_sort) + ".png"
    cv2.imwrite(filename, img)
    print("saved: " + filename)



def bruteforce(target, folder):
    for n_cluster in n_clusters:
        classifier = ImageClassifier(n_cluster, target)
        for color_in in colorspaces:
            df = classifier.get_dataframe(colorspace=color_in)
            for key in kmeans_keys:
                cluster_map = classifier.run_kmeans(df, key)
                clusters = classifier.get_clusters(cluster_map)
                clusters_bak = clusters.copy()
                for color_sort in colorspaces:
                        for sorting_key in sorting_keys:
                            cmp1 = sorting_lambdas[sorting_key]
                            cmp2 = cl_sorting_lambdas[sorting_key]
                            clusters = classifier.sort_clusters(clusters, cmp1, color_sort=color_sort)
                            res = classifier.merge_clusters(clusters, cmp2)
                            save(folder, res, n_cluster, key, color_in, sorting_key, color_sort)
                            clusters = clusters_bak.copy()



def process():
    n_cluster = 4
    classifier = ImageClassifier(n_cluster, 'src.jpg')
    df = classifier.get_dataframe(colorspace=cv2.COLOR_BGR2HSV)
    cluster_map = classifier.run_kmeans(df, [0])
    clusters = classifier.get_clusters(cluster_map)

    clusters_bak = clusters.copy()

    #cmp = lambda pixel: (255 - int(pixel[1])) * 2 - (200 if pixel[1] < pixel[2] else 0)
    cmp = lambda pixel: int(pixel[0])
    #cmp = lambda pixel: pixel[1]
    clusters = classifier.sort_clusters(clusters, cmp, color_sort=cv2.COLOR_BGR2LAB)
    res = classifier.merge_clusters(clusters, lambda cluster: sum(cluster[0][0]))
    #filename = 'res_sort/res_{0}_{1}_{2}.png'.format(coeff[0], coeff[1], coeff[2])
    filename="res.png"
    cv2.imwrite(filename, res)
    print('saved {0}'.format(filename))

    clusters = clusters_bak.copy()



def compare(target1, target2):

    cl1 = ImageClassifier(4, target1)
    cl2 = ImageClassifier(4, target2)

    df1 = cl1.get_dataframe()
    df2 = cl2.get_dataframe()

    print(df1.describe())
    print(df2.describe())
    exit()
    img1 = cv2.imread(target1)
    img2 = cv2.imread(target2)
    shape1 = img1.shape
    shape2 = img2.shape

    img1 = np.reshape(img1, (shape1[0] * shape1[1], 3))
    img2 = np.reshape(img2, (shape2[0] * shape2[1], 3))

    img1 = sorted(img1, key = lambda pixel: sum(pixel))
    img2 = sorted(img2, key = lambda pixel: sum(pixel))
    
    img1 = np.reshape(img1, (shape1))
    img2 = np.reshape(img2, (shape2))

    cv2.imwrite('img1.png', img1)
    cv2.imwrite('img2.png', img2)


# bruteforce("town.jpg", "result/town")
# compare("res.png", "town.jpg")
process()
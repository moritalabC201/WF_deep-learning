import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import os
from os.path import join as pjoin
from wfield.utils import point_find_ccf_region
from typing import List, Tuple, Union
from collections import namedtuple
IndependentComponentInfo = namedtuple("IndependentComponentInfo", 
                                      ["id_list", "subid_list", "centroid_list", "areaname_list", "acronym_list"])

class IndependentComponent(object):

    ## This object is initiated with the minimum information, self.image, with which the curation analysis starts.
    def __init__(self, image:np.ndarray[float], mask:np.ndarray=None, id:str=None, threshhold:float=0.975):
        
        # These three entry is filled when the instance is initiated
        H, W = image.shape
        self.image = image
        if mask is None:
            mask = np.ones(image.shape)
            self.zimage = self.__set_zimage(mask)
        else:    
            self.zimage = self.__set_zimage(mask)
        self.bimage = self.__set_bimage(threshhold)
        self.id = id # This is just for seeing where each independent component object comes from
        self.subid = None
        self.ncomponents = None # this field is set only for the independent componentss that were splited from a single connected independent component 
        self.areasize = None # After applying spatial filter to binarized image, this value is set only to ics, not to the ancestors of them
        self.coordinate_of_centroid = None # After applying spatial filter to binarized image, this value is set only to ics, not to the ancestors of them
        self.curated_image = None # This is the true independent component image, after applied to the final bitmask
        
    def __set_zimage(self, mask:np.ndarray)->np.ndarray[float]:
        average = np.nanmean(mask*self.image)
        stdev = np.nanstd(mask*self.image) + 1e-8
        return (mask*self.image - average) / stdev 
    
    def __set_bimage(self, threshhold:float)->np.ndarray:

        def _find_first_exceeding_index(sorted_list, threshold):
            for i, value in enumerate(sorted_list):
                if value > threshold:
                    return i
            return -1  # If no element exceeds the threshold
        
        if not (0< threshhold < 1):
            raise ValueError("Threshold should be more than 0, and leaa than 1.\n")

        z = self.zimage.copy().flatten()
        N = len(z[~np.isnan(z)])
        z = z.flatten()[~np.isnan(z)]
        sx = sorted(z)
        sy = [i/N for i in range(N)]
        index_sy = _find_first_exceeding_index(sorted_list=sy, threshold=threshhold)
        threshold_z = sx[index_sy]
        return np.where(self.zimage > threshold_z, 1, 0).astype("uint8")
    
    def get_image(self):
        return self.image
    
    def get_zimage(self):
        return self.zimage
    
    def get_bimage(self):
        return self.bimage
    
    def get_curated_image(self):
        return self.curated_image
    
    def get_id(self):
        return self.id
    
    def get_subid(self):
        return self.subid
        
    def get_ncomponents(self):
        return self.ncomponents
    
    def get_centroid(self):
        return self.coordinate_of_centroid
    
    def update_fields(self, bitmap:np.ndarray, ncomponents:int=1, centroid:Tuple[int,int]=None, subid:str=None):
        
        # update info for filtering
        self.subid = subid
        self.ncomponents = ncomponents

        # get the final curated image applying the bitmap, which is provided in another function
        self.curated_image = np.squeeze(bitmap * self.image)
        self.areasize = np.sum(bitmap)
        self.coordinate_of_centroid = centroid 

    

def split_independent_component(ic:IndependentComponent,
                                areasize_threshold:int=2000)->List[IndependentComponent]:
    
    '''
    This function aims to curate independent component image and returns a list that contains independent component object
    1. independent component image is curated out -> blank, empty list is returned
    2. independent component image is curated and just a single independent component is remained -> list that contains the updated input ic is returned
    3. independent component image is curated and more than one independent component is detected -> list of all of the independent components detected is returned
    '''

    '''
    nlabels : integer, number of connectedComponents found
    labels : numpy.ndarray object, each detected connected component area has the same number
    stats[:, -1] : numpy.ndarray object, with shape (nlabels, 1), stores areasize info
    centroids : numpy.ndarray object. with shape (nlabels, 2), each row stores for (x, y) of centroid
    '''
    import copy
    from cv2 import connectedComponentsWithStats
    nlabels, labels, stats, centroids = connectedComponentsWithStats(ic.get_bimage())

    # make mask for extracting each separated area larger than pixel_area_threshold (in a independent component image)
    areasize_for_labels = stats[:,-1] 
    masks = areasize_for_labels > areasize_threshold 
    masks = masks[1:] # Background (whole 0 frame, with size of H*W) is also included, that is why removing the first element in the mask
    centroids = centroids[1:] # centroid for background is neither needed

    # Still, you cannot extract each separated area in a independent component image. You just have a single mixed np.ndarray.
    # Here we get bitmaps for each connectedComponent
    components = []
    for label in range(1, nlabels):  # ignore 0, as that means the background and nothing will be returned.
        component = (labels == label).astype('uint8')  # component is the bitmap for each detected connectedComponent
        components.append(component)

    # convert object type into numpy.ndarray with shape (nlables, H, W)
    # Each (H, W) frame is bit map for extracting each separated area in a independent component image
    components = np.array(components) 

    # Not all the bitmap is needed, as some of them is smaller than (<) the areasize_threshold
    bitmaps = components[masks, :, :]
    centroids = centroids[masks, :]

    '''
    From here in this function implementation is the critical point.
    Depending on how many connenctedComponets detected, different processing is needed.
    - Just one : Straightfoward
    - More than one : need to create new IndependentComponent instaces
    '''

    H, W = ic.get_bimage().shape
    list_to_return = []

    # Depending on the length of bitmap, you need to change the program here.
    # length = 0 ; no loop will be done, so self.curated_image is not updated and None -> should be filtered after
    # length = 1 ; just one independent component
    # length = 2 ; ic is to be splitted

    if len(bitmaps) == 0: # Nothing left after spatial filtering, -> delete
        return list_to_return
    
    else:
        '''
        The code block just below is unnecessary. If no problem occurs, just comment out below. 
        The code below includes the original independent component images that has multiple regions, each of which has been splitted into a single region.
        '''
        # Append the original, if only one independent component is included, that case is done here

        if len(bitmaps) == 1:
            ic.update_fields(bitmap=bitmaps, ncomponents=1, centroid=centroids, subid=ic.get_id())
            list_to_return.append(ic)
            return list_to_return


        if len(bitmaps) >= 2:
            all_bitmap = np.zeros((H,W))
            loop_count = 0
            for bitmap in bitmaps:
                all_bitmap += bitmap
                loop_count += 1
            
            connected_ic = copy.copy(ic)
            connected_ic.update_fields(bitmap=all_bitmap, ncomponents=loop_count, centroid=None, subid=ic.get_id())
            list_to_return.append(connected_ic)

            for i, (bitmap, centroid) in enumerate(zip(bitmaps, centroids), start=1):
                separate_ic = copy.copy(ic)
                separate_ic.update_fields(bitmap=bitmap, ncomponents=1, centroid=centroid, subid=ic.get_id() + "_#" + str(i))
                list_to_return.append(separate_ic)       
        
        return list_to_return


def filter_independent_components(ics:List[IndependentComponent])->Union[List[IndependentComponent], List[IndependentComponent]]:
    '''
    This function filters independent components object in a list into two lists, and return them as an tuple object
    mixed_ics is a list object that contains independent component object of
    '''
    mixed_ics = []
    splitted_ics = []
    
    for ic in ics:
        if ic.get_ncomponents() == 1: # ic.get_ncomponents() >= 1, if anything unexpected happens might be None
            # if ncomponents == 1, then that should belong to both
            splitted_ics.append(ic) 
            mixed_ics.append(ic)
        else:
            # Given that ic.get_ncomponents() >= 1, if ncomponents != 1, then ncomponents >=2 and should belong to mixed_ics
            mixed_ics.append(ic) # independent components that has single region only
    
    return mixed_ics, splitted_ics

def visualize_images(ic:IndependentComponent,
                     mask:np.ndarray=None,
                     ccf_regions = None,
                     magnify:float=1.5)->None:

    '''
    This function is just a wrapper function, actual visualization is done by __visualize()
    Set necessary parameter and pass them to t__visualize()
    '''

    def __calculate_image_range(image:np.ndarray)->Tuple[float, float]:
        """
        Calculate the minimum and maximum values of an image.

        Parameters:
        - image (numpy.ndarray): Input image.

        Returns:
        - Tuple (min_value, max_value): Minimum and maximum values of the image.
        """
        min_value = np.min(image)
        max_value = np.max(image)
        return (min_value, max_value)

    # Example usage:
    # Suppose you have an image called 'my_image'.
    # Call the function as follows:
    # min_val, max_val = calculate_image_range(my_image)
    # print(f"Minimum value: {min_val}, Maximum value: {max_val}")


    import matplotlib.pyplot as plt

    if mask is None:
        H, W = ic.get_image().shape
        mask = np.ones((H, W))
    def __visualize(images:List[np.ndarray],
                    value_ranges:List[Tuple],
                    titles:List[str],
                    multiply=magnify):
        '''
        images = [image1, image2, image3, ...]
        value_ranges = [(min, max), (min, max), ....]
        '''
        num_images = len(images)

        # Display all images in a single row
        rows = 1
        cols = num_images

        fig, axes = plt.subplots(rows, cols, figsize=(8*multiply, 6*multiply))
        fig.subplots_adjust(wspace=0.5)

        for i, title in zip(range(num_images), titles):
            ax = axes[i]
            img = images[i]

            # Normalize the image to the range [0, 1]
            img_normalized = (img - value_ranges[i][0]) / (value_ranges[i][1] - value_ranges[i][0])

            im = ax.imshow(mask*img_normalized, cmap='viridis')  
            cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.05, pad=0.05)
            ax.set_title(f'{title}')

            if ccf_regions is not None:
                for c in ccf_regions.iterrows():
                    c = c[1]

                    one_region_r = np.vstack([np.array(c.right_x), np.array(c.right_y)]).T
                    ax.plot(one_region_r[:,0], one_region_r[:,1], color="white", lw=0.5)
                    
                    one_region_l = np.vstack([np.array(c.left_x), np.array(c.left_y)]).T
                    ax.plot(one_region_l[:,0], one_region_l[:,1], color="white", lw=0.5)

        plt.show()

    # call the __visualize() function
    # returns curated_image, if they are already calculated, probably after curation process
    if ic.get_curated_image() is None:
        images = [ic.get_image(), ic.get_zimage(), ic.get_bimage()]
        value_ranges = [__calculate_image_range(ic.get_image()), (-2,2), (0,1)]
        titles = ["After JADE", "Z-scored", "Thresholded"]
        __visualize(images, value_ranges, titles)
    else:
        images = [ic.get_image(), ic.get_zimage(), ic.get_bimage(), ic.get_curated_image()]
        value_ranges = [__calculate_image_range(ic.get_image()), (-2,2), (0,1), __calculate_image_range(ic.get_curated_image())]
        titles = ["After JADE", "Z-scored", "Thresholded", "Curated"]
        __visualize(images, value_ranges, titles)


def save_curated_images_and_ids(ics:List[IndependentComponent],
                                identifier:str,
                                path:str,
                                ccf_region:pd.DataFrame)->Union[List[np.ndarray], List[Tuple]]:

    import pickle

    # list to which append the related info
    curated_images = []
    ids = []
    subids = []
    centroids = []
    areanames = []
    acronyms = []

    for ic in ics:
        image = ic.get_curated_image()
        if curated_images is not None:
            curated_images.append(image)
            ids.append(ic.get_id())
            subids.append(ic.get_subid())
            centroids.append(ic.get_centroid())
            centroid = ic.get_centroid().squeeze().tolist()
            # print(type(centroid))
            # print(len(centroid))
            # print(centroid)
            centroid_region_info, tmp, _ = point_find_ccf_region(centroid, ccf_region)
            areanames.append(centroid_region_info["name"])
            acronyms.append(centroid_region_info["acronym"])
    
    info = IndependentComponentInfo(id_list = ids,
                                    subid_list = subids,
                                    centroid_list=centroids,
                                    areaname_list=areanames,
                                    acronym_list=acronyms)

    curated_images = np.stack(curated_images) # convert list object into np.ndarray object
    ncomponents, H, W = curated_images.shape # get the shape of the np.ndarray object
    curated_matrix = normalize(curated_images.reshape((ncomponents, H*W))) # reshape np.ndarray object 
    
    ic_matrix_savepath = pjoin(path, identifier + "_" + "ic_matrix" + ".npy")
    ic_image_savepath = pjoin(path, identifier + "_" + "ic_image" + ".npy")
    info_savepath = pjoin(path, identifier + "_" + "ic_info" + ".pkl")

    np.save(ic_matrix_savepath, curated_matrix)
    np.save(ic_image_savepath, curated_images)
    with open(info_savepath, 'wb') as file:
        pickle.dump(info, file)


def load_info(path):
    import pickle
    with open(path, 'rb') as file:
        info = pickle.load(file)
    print(f"These are the field values, which is the related info of each ICs.")
    print(f"Each field is acessible by dot style")
    print(f"{info._fields}")
    return info


'''
The scripts below are old legacy codes.
'''

# import numpy as np
# def z_score(ic_image:np.ndarray)->np.ndarray:
#     average = np.mean(ic_image)
#     stdev = np.std(ic_image) + 1e-25
#     ic_image_z = (ic_image - average) / stdev 
#     return ic_image_z


# def apply_mask(ic_images_z:np.ndarray, threshhold:np.float16)->np.ndarray:
#     return np.where(ic_images_z > threshhold, 1, 0)


# def separate_ics(thresh_holded_ic_images:np.ndarray,
#                  ic_images_masked_bin_uint8:np.ndarray, 
#                  pixel_area_threshold:int):
    
#     import copy
#     import cv2
    
#     # get image shape 
#     # H, W = ic_images_masked_bin_uint8[0].shape

#     # list for storing each separated ICs
#     connected_ICs = []
#     connected_ICs_size = []
#     separate_ICs = []
#     separate_ICs_id = []
#     separate_ICs_size = []
#     centroids = []
    
#     for index, (image, image_original) in enumerate(zip(ic_images_masked_bin_uint8, thresh_holded_ic_images)):
#         # index_ic is to indicate which ic you are processing in this loop
#         # image is z-scored, threshholded, masked, binarized and converted explicitly to uint8
#         # image_original is each ICs extracted directly by ICA-JADER algorithm
#         retval, labels, stats, centroid = cv2.connectedComponentsWithStats(image)

#         # make masks for extracting each separated area in a independent component image, 
#         # whose area is larger than pixel_area_threshold
#         # Background (whole 0 frame) is also extracted here, that is why remove the first element in the mask
#         pixel_area_size = stats[:,-1]
#         masks = pixel_area_size > pixel_area_threshold
#         masks = masks[1:]

#         # At this point, you cannot extract each separated area in a independent component image
#         # Here we get bitmaps for that purpose
#         # components is (ncomponents_separatedArea, H, W)
#         # Each (H, W) frame is bit map for extracting each separated area in a independent component image
#         components = []
#         for label in range(1, retval):  # 0は背景なので無視
#             component = (labels == label).astype('uint8')  # ラベルに対応する成分を抽出
#             components.append(component)

#         components = np.array(components)

#         # iterate along the first axis of component and apply the bitmap to the original ic_images,
#         # Then you can get each separated area in single IC image
#         bitmaps = components[masks, :, :]

#         # bitmapsの長さが0の時には、for blockすべてスキップなので、separated_ICは変更なし
#         # IC_keysのほうも、何も追加されないので、ばらされた各ICとおおもとのICの対応がつく

#         # extract each separated region in single ic_image
#         key = "IC" + str(index+1)
#         for i, bitmap in enumerate(bitmaps, start=1):
#             separated_IC = image_original * bitmap
#             separate_ICs.append(separated_IC)
#             separate_ICs_size.append(int(np.sum(bitmap)))
#             if len(bitmaps) <= 1:
#                 separate_ICs_id.append(key)
#             else:
#                 separate_ICs_id.append(key + "#" + str(i))
                    

#         # key_str = "IC" + str(index+1)
#         # key_array = np.repeat(key_str, len(bitmaps))
#         # IC_keys.extend(key_array.tolist())

#         image_original_copy = image_original.copy()
#         if len(bitmaps) == 0:
#             all_bitmap = np.zeros(shape=components[0].shape)
#             connected_ICs_size.append(int(np.sum(all_bitmap)))
#         else:
#             all_bitmap = np.zeros(shape=components[0].shape)
#             for bitmap in bitmaps:
#                 all_bitmap += bitmap
#             connected_ICs_size.append(int(np.sum(all_bitmap)))
                
#         connected_ICs.append(image_original_copy*all_bitmap)
#         # connected_ICs_size.append(int(np.sum(all_bitmap)))
        
#     separate_ICs = np.array(separate_ICs)
#     separate_ICs_id = np.array(separate_ICs_id)
#     separate_ICs_size = np.array(separate_ICs_size)
#     connected_ICs = np.array(connected_ICs)
#     connected_ICs_size = np.array(connected_ICs_size)
#     # centroids = np.arange(centroids)
    
#     return separate_ICs, separate_ICs_id, separate_ICs_size, connected_ICs, connected_ICs_size

# def plot_ic_images(image_matrix:np.ndarray, arrange_shape:tuple, colorbar=False):

#     import matplotlib.pyplot as plt
#     row, col = arrange_shape
#     fig, axs = plt.subplots(row, col, figsize = (16, 12))
#     axs = axs.reshape(-1)

#     for idx, ax in enumerate(axs):
#         data = image_matrix[idx]
#         max_pixel_value = np.max(data)
#         min_pixel_value = -np.max(data)
#         heatmap = ax.imshow(data, cmap='viridis', vmin=min_pixel_value, vmax=max_pixel_value)
#         ax.set_title("IC" + str(idx+1))
    
#     # Add a single colorbar
#     if colorbar:
#         fig.subplots_adjust(right=0.8)
#         cbar_ax = fig.add_axes([1, 0.0, 0.05, 0.7])
#         cbar = fig.colorbar(heatmap, cax=cbar_ax)

#     plt.tight_layout()
#     plt.show()
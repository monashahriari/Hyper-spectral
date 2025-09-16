# import necessary packages
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
import os
from glob import glob


class ChangeDetection:
    def __init__(self,before,after,gt, data_name,thr_dict):
        """
        Initialize the ChangeDetection instance with before and after images and ground truth.

        Parameters:
        - before: np.ndarray, image before change (H x W x D)
        - after: np.ndarray, image after change (H x W x D)
        - gt: np.ndarray, ground truth change map (H x W)
        - data_name: str, name of the dataset for labeling
        """
        self.before = before
        self.after = after
        self.gt = gt 
        self.data_name = data_name
        self.h , self.w , self.d = self.before.shape
        self.thr_dict = thr_dict

    def normalization(self):
        """
        Normalize the before and after images using z-score normalization across all bands.

        Returns:
        - before, after: Normalized flattened images (N x D)
        """
        before = self.before.reshape(self.h*self.w,-1)
        after = self.after.reshape(self.h*self.w,-1)
        scaler = StandardScaler()
        scaler.fit(np.vstack([before, after]))
        before = scaler.transform(before)
        after = scaler.transform(after)
        return before , after

    def sam(self,before,after):
        """
        Apply Spectral Angle Mapper (SAM) for change detection.

        Parameters:
        - before, after: Normalized image vectors (H*W x D)

        Returns:
        - sam: Binary change map (H*W x 1)
        - angle: Computed angle values in radians (H*W x 1)
        """
        sam = np.zeros((self.h*self.w,1))
        angle = np.zeros((self.h*self.w,1))
        
        for i in range(before.shape[0]):
            dot_product = np.dot(before[i,:],after[i,:])
            norm1 = np.linalg.norm(before[i,:])
            norm2 = np.linalg.norm(after[i,:])
            angle[i,0] = np.arccos(np.abs(dot_product / (norm1 * norm2)))
    
            if angle[i,0] > self.thr_dict['sam_thr']:
                sam[i,0] = 1
            else:
                sam[i,0] = 0
        
        return sam, angle

    def ID(self,before,after):
        """
        Perform Image Difference (ID) method for change detection using Euclidean distance.

        Parameters:
        - before, after: Normalized image vectors (H*W x D)

        Returns:
        - id_matrix: Pixel-wise difference vector (H*W x D)
        - binary_change_from_id: Binary change map (H*W x 1)
        """
        id_matrix = np.zeros_like(before)
        binary_change_from_id = np.zeros((self.h*self.w,1))
        change_2d = np.zeros((self.h*self.w,1))

        for i in range(id_matrix.shape[0]):
            for j in range(id_matrix.shape[1]):
                id_matrix [i,j] = after[i,j] - before[i,j]

            change_2d[i,0]= np.linalg.norm(id_matrix[i,:])

            if change_2d[i,0] >=self.thr_dict['id_thr']:

                binary_change_from_id[i,0] = 1
            else:
                binary_change_from_id[i,0] = 0

        return id_matrix,binary_change_from_id
    
    def modified_z_score(self,image_diff):
        """
        Compute the Modified Z-score (ZDI) of the image difference.

        Parameters:
        - image_diff: Difference image (H*W x D)

        Returns:
        - normalized_zscore: Normalized ZDI map (H*W x 1)
        """
        num_pixels, num_bands = image_diff.shape
        normalized_zscore = np.zeros((num_pixels,))
        
        for i in range(image_diff.shape[1]):
            mean = np.mean(image_diff[:,i])
            std = np.std(image_diff[:,i])
            z_squared  = ((image_diff[:,i] - mean)/std)**2
            normalized_zscore += z_squared 
        
        normalized_zscore = (normalized_zscore - np.min(normalized_zscore))/(np.max(normalized_zscore) - np.min(normalized_zscore))
        normalized_zscore = normalized_zscore.reshape(self.h*self.w,1)
        return normalized_zscore
    
    def sam_zdi(self,angle,normalzdi):
        """
        Combine SAM angle with normalized ZDI using tangent and sine functions.

        Parameters:
        - angle: SAM angle matrix (H*W x 1)
        - normalzdi: Normalized ZDI values (H*W x 1)

        Returns:
        - sam_zdi_sin, sam_zdi_tan: Binary change maps (H*W x 1)
        """
        sam_zdi_tan = np.zeros_like(angle)
        sam_zdi_sin = np.zeros_like(angle)
        for i in range(angle.shape[0]):
            
            sam_zdi_tan[i,0] = np.tan(angle[i,0])*normalzdi[i,0]
            sam_zdi_sin[i,0] = np.sin(angle[i,0])*normalzdi[i,0]
        
        for i in range(sam_zdi_tan.shape[0]):
            sam_zdi_sin[i,0] = 1 if sam_zdi_sin[i,0] >= self.thr_dict['zdi_sin_thr'] else 0
            sam_zdi_tan[i,0] = 1 if sam_zdi_tan[i,0] >= self.thr_dict['zdi_tan_thr']  else 0

        return sam_zdi_sin,sam_zdi_tan

    def plot_predict(self,result,result_name):
        """
        Plot the prediction result, ground truth, and their difference. Also displays metrics and confusion matrix.

        Parameters:
        - result: Binary prediction result (H*W x 1)
        - result_name: str, method name
        """
        gt = self.gt.reshape(-1,1)

        cm = metrics.confusion_matrix(gt, result, labels=[0,1])
        # calculate metrics
        ACC_T = (cm[0,0] + cm[1,1]) / np.sum(cm)
        IOU_T = cm[0,0] / (cm[0,0] + cm[0,1] + cm[1,0])
        TPR_T = cm[0,0] / (cm[0,0] + cm[1,0])
        FPR_T = cm[0,1] / (cm[0,1] + cm[1,1])
        print(f'\n{result_name} result: acc: {ACC_T:.3f} , IOU: {IOU_T:.3f}\n')

        # difference map
        dif = gt - result
        dif_2d = dif.reshape(self.h, self.w)

        # figure with custom gridspec
        f = plt.figure(self.data_name)
        gs = gridspec.GridSpec(1, 3, width_ratios=[4,4, 10])  # third plot wider

        plt.suptitle(f'{result_name} on {self.data_name}')

        # legend for 0/1
        legend_elements = [
            Patch(facecolor='black', edgecolor='black', label='0 = No change'),
            Patch(facecolor='white', edgecolor='black', label='1 = Change')
        ]

        # Subplot 1: result
        ax1 = f.add_subplot(gs[0])
        ax1.imshow(result.reshape(self.h, self.w), cmap='gray')
        ax1.set_title(f'{result_name} result')
        ax1.axis('off')

        # Subplot 2: GT
        ax2 = f.add_subplot(gs[1])
        ax2.imshow(gt.reshape(self.h, self.w), cmap='gray')
        ax2.set_title('GT')
        ax2.axis('off')
        ax2.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.5, -0.1), ncol=2)

        # Subplot 3: Difference (larger)
        ax3 = f.add_subplot(gs[2])
        im = ax3.imshow(dif_2d, cmap='bwr')
        ax3.set_title('difference')
        ax3.axis('off')

        # custom legend for difference
        labels = ['alpha', 'true', 'betta']
        values = np.unique(dif_2d.ravel())
        colors = [im.cmap(im.norm(value)) for value in values]
        patches = [mpatches.Patch(color=colors[i], label=f'{labels[i]}') for i in range(len(values))]
        ax3.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        # Table
        columns = ['IOU', 'ACC', 'TPR', 'FPR']
        row = ['value']
        list = ["%.2f" % IOU_T, "%.2f" % ACC_T, "%.2f" % TPR_T, "%.2f" % FPR_T]
        cell_text = [list]
        the_table = plt.table(cellText=cell_text, rowLabels=row, colLabels=columns, loc='bottom')
        # Make table bigger
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(10)
        the_table.scale(1.5, 1.5)
        plt.show()


    def main(self):
        """
        Main function to run all change detection methods and plot their predictions.
        """
        normalized_before,normalized_after = self.normalization()
        sam,angle = self.sam(normalized_before,normalized_after)
        id_mat,binary_change = self.ID(normalized_before,normalized_after)
        zdi = self.modified_z_score(id_mat)
        zdi_sin , zdi_tan = self.sam_zdi(angle,zdi)
        self.plot_predict(sam,'SAM')
        self.plot_predict(binary_change,'Image difference')
        self.plot_predict(zdi_sin,'zdi_sin')
        self.plot_predict(zdi_tan,'zdi_tan')

# Load data from .mat file and run change detection
root = os.getcwd()
data_path = os.path.join(root,"..","Data")

usa_path = glob(os.path.join(data_path,"*.mat"))[0]
data_dict = {'USA':{'sam_thr':1.25,'id_thr':10,'zdi_sin_thr':0.05,'zdi_tan_thr':0.25,'path':usa_path}}
for name,value in data_dict.items():
    data = sio.loadmat(value['path'])
    before, after, gt  = data['T1'], data['T2'], data['Binary']

    # create instance
    chnage_detection = ChangeDetection(before,after,gt,name,value)
    chnage_detection.main()



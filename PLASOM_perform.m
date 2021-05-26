%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     Gardens2 algorithm                                                  %
%     Jonás Grande Barreto                                                %
%     María Del Pilar Gómez Gil                                           %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;clc
close all
% NIfTI_files package can be found at
% https://www.mathworks.com/matlabcentral/fileexchange/8797-tools-for-nifti-and-analyze-image)
% addpath(('NIfTI_files'))

% Range of scans where the brain is located a standard brain MRI volume 
% contains 200 scans. The brain is usually located in the middle of the 
% MRI volume
rt =[63,201];  %<---example  
sample = 1;

tissues = 3; 
MS = [4,6,8,10,12,16,20];
S = 4;

mapsize = [MS(S),MS(S)]; 
node_row = mapsize(1); 
node_col = mapsize(2); 

map1 = [ 1   1   1      
        0.9 0.9  0    
        0.0 0.1 0.9    
        0.0 0.9 0.0     
         0.9  0  0.1      
         1   0   1];    

ACC = zeros(2,3);

%% Load files
load(['SUPSOM_fuzzy_class_GRDNS2U',num2str(node_row),'x',...
        num2str(node_col),'_3folds_N4itk.mat'],'MiSom')
LABELp1p2 = MiSom.SOM_LBLp1p2; 
LABELp1p2f = MiSom.SOM_LBLp1p2; 
brain_msk = load_nii('IBSR_01_ana_brainmask.nii');  
GT = load_nii(['IBSR_01_segTRI_fill_ana.nii']);  
%    load('Gardens2 pro\feats_Subject_x.mat','feat_all_ibsr')  

%% Index to adjoining scans   
[col,row,dip] = size(brain_msk.img);
midd = round((rt(2)-rt(1))/2)+rt(1);
slix = midd- 20 : midd + 29;
point = rt(1) : rt(2);
scan_length =length(slix);
   
Clust_imgp2 = zeros(row,col,length(slix));
Clust_imgp1p2 = zeros(row,col,length(slix));
Taux = zeros(scan_length,3);
for scan = 1 : scan_length
    datap1p2f = zeros(row*col,1); 
    datap1p2 = zeros(row*col,1);
    sliceg = slix(scan); 
    mask = imrotate(logical(brain_msk.img(:,:,sliceg)),90);
    gt = imrotate((GT.img(:,:,sliceg)),90);
    xs = find(mask); 

%     gt = imrotate((GT.img(:,:,sliceg)),90); 
    datap1p2f(xs) = LABELp1p2f(1:length(xs));    
    datap1p2(xs) = LABELp1p2(1:length(xs));    
    LABELp1p2f((1:length(xs))) = [];
    LABELp1p2((1:length(xs))) = [];
    SOM_clup1p2f = reshape(datap1p2f,row,col);   
    SOM_clup1p2 = reshape(datap1p2,row,col);   

        
    [FP,FN,TP,TN] = confuzzt(SOM_clup1p2,gt,tissues);
    SOM.AOMp1p2(:,scan,sample) = round((2*TP)./(2.*TP+FP+FN),3,'significant');
    Clust_imgp1p2(:,:,scan) = SOM_clup1p2f;
    Taux(scan,:) = SOM.AOMp1p2(:,scan,sample)';
        
    subplot(1,2,1)
    imshow(SOM_clup1p2f,[],'InitialMagnification','fit')
    title (['Resutl ',num2str(sliceg)])
    subplot(1,2,2)
    imshow(uint8(gt),map1,'InitialMagnification','fit')
    title (['GT ',num2str(scan)])
    pause (1)
            
end
% 
 checkk= [mean(Taux,1), std(Taux,0,1)]
% 
function [FP,FN,TP,TN] = confuzzt(fg,gt,tissue)
FP = zeros(tissue,1);
FN = zeros(tissue,1);
TP = zeros(tissue,1);
TN = zeros(tissue,1);
fg_inx = [1,2,3];
    for k = 1 : tissue
%         a = fg == k;
        a = (fg == fg_inx(k));
        A = gt == k;
        a = logical(a);
%             c = a & TG;
        d = a & A;
        FP(k) = sum(sum(a&~A));
        FN(k) = sum(sum(~a&A));
        TP(k) = sum(sum(a&A));
        TN(k) = sum(sum(~a&~A));
    end
end
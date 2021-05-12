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
addpath(('NIfTI_files'))

% Range of scans where the brain is located a standard brain MRI volume 
% contains 200 scans. The brain is usually located in the middle of the 
% MRI volume
rt =[63,201];  %<---example  
sample = 1;
% Level of mixture between two classes
pveI= 0.20;


map1 = [ 1   1   1    
        0.9 0.9  0   
        0.0 0.1 0.9    
        0.0 0.9 0.0    
         0.9  0  0.1   
         1   0   1];   
     
SOMG_grdn = [];

figure

%% Load files
brain_msk = load_nii('IBSR_01_ana_brainmask.nii');  
% load('Gardens2_hard_output.mat','zave','CI')
% load('G2_partial_maps_NL_N4ITK.mat','csf_u','gm_u','wm_u')
load('G2_partial_maps_I_N4ITK.mat','csf_u','gm_u','wm_u')
 %% Index to adjoining scans            
[col,row,dip] = size(brain_msk.img);    
midd = round((rt(2)-rt(1))/2)+rt(1);
slix = midd- 20 : midd + 29;
point = rt(1) : rt(2);
scan_length =length(slix);

PL = []; %<---pseudo labels
for scan = 1 :  scan_length
        %% Bloque 1
        plabels = zeros(row,col);
        hlabels = zeros(row,col);
        sliceg = slix(scan); 
        xslice = find(point == sliceg);
        mask = double(imrotate(round(abs(brain_msk.img(:,:,sliceg))),90)); 
        
        csf = (csf_u(:,:,scan));
        gm = (gm_u(:,:,scan));
        wm = (wm_u(:,:,scan));
        
        xs = find(mask); 
        % pseudolabel computation
        for ix = 1 : length(xs)
            n = xs(ix);
            if (csf(n) >= pveI) && (gm(n) >= pveI)
                plabels(n) = 4;
            elseif (gm(n) >= pveI) && (wm(n) >= pveI)
                plabels(n) = 5;
            else
                [~,dex] = max([csf(n),gm(n),wm(n)]);
                plabels(n) = dex;
            end
            [~,dex] = max([csf(n),gm(n),wm(n)]);
            hlabels(n) = dex;
        end
        
        % number of scan
        sli_num=ones(row*col,1)*sliceg;
        % number of volume
        pak_num=ones(row*col,1)*sample;
        

%         if scan == 30
%             close all
%             figure, imshow(uint8(plabels),map1,'InitialMagnification','fit')
%             break
%         end
      

        DU = [pak_num,sli_num,hlabels(:),plabels(:)];
        DUa = DU(xs,:);
        PL = [PL;DUa];

        formatSpec = 'Sujeto..%u...Slice %u/%u ...%3.3f%% \r';
        A1 = (scan/length(slix))*100;
        fprintf(formatSpec,sample,scan,length(slix),A1)
        
        subplot(1,2,1)
        imshow(uint8(plabels),map1,'InitialMagnification','fit')
        title (['Partial map scan ',num2str(sliceg)])
        subplot(1,2,2)
        imshow(uint8(hlabels),map1,'InitialMagnification','fit')
        title (['Hard map scan ',num2str(sliceg)])
        pause (0.01)
    end

 save (['PseudoLabels_index_',num2str(pveI*100),'_nbf.mat'],'PL')



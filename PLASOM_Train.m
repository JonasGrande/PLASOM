%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     Gardens2 algorithm                                                  %
%     Jonás Grande Barreto                                                %
%     María Del Pilar Gómez Gil                                           %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;clc
close all

% somtoolbox package can be found at
% http://www.cis.hut.fi/somtoolbox/about.shtml
addpath(('somtoolbox'))

% NIfTI_files package can be found at
% https://www.mathworks.com/matlabcentral/fileexchange/8797-tools-for-nifti-and-analyze-image)
addpath(('NIfTI_files'))

% Range of scans where the brain is located a standard brain MRI volume 
% contains 200 scans. The brain is usually located in the middle of the 
% MRI volume
rt =[63,201];  %<---example  
sample = 1;

% Size of the SOM
MS = [4,6,8,10,12,16,20];
S = 4; 
tissues = 3;
expo = 2;
pveI = 0.2;
mapsize = [MS(S),MS(S)]; 
node_row = mapsize(1);
node_col = mapsize(2); 
maximo_clases = tissues;

total_clusters = node_row*node_col;

% internal parameters
Beta = 0.07;
Gama = 0.05;
wnw3 = 3;
d3=(wnw3-1)/2;


    %% Load files
    brain_msk = load_nii('IBSR_01_ana_brainmask.nii');  
    atlas_csf = load_nii('csf_template.nii');
    atlas_gm  = load_nii('gm_template.nii');
    atlas_wm  = load_nii('wm_template.nii'); 
    load('Gardens2 pro\feats_Subject_x_N4itk.mat','feat_all_ibsr')
    load('G2_partial_maps_N4itk.mat','csf_u','gm_u','wm_u')
    load (['PseudoLabels_index_',num2str(pveI*100),'_N4itk.mat'],'PL')
    
    csf_map = csf_u;
    gm_map = gm_u;
    wm_map = wm_u;
    
    %% Index to adjoining scans   
    [col,row,dip] = size(brain_msk.img);
    midd = round((rt(sample,2)-rt(sample,1))/2)+rt(sample,1);
    slix = midd- 20 : midd + 29;
    point = rt(1) : rt(2);
    scan_length =length(slix);
    XS = [];
    data = zeros(row*col,15); 
    DATA = [];
    
    for scan = 1 : scan_length  
        sliceg = slix(scan);
        xslice = find(point == sliceg);
        mask = double(imrotate(round(abs(brain_msk.img(:,:,sliceg))),90)); %      

        ai = d3+1; af = row +d3;
        bi = d3+1; bf = col+d3;
        IM=(zeros(row+(2*d3),col+(2*d3))); 
        IM(ai:af,bi:bf) = mask; 
        xs = find(IM);
        XS = [XS;xs];
        
        data(xs,:) = feat_all_ibsr(1:length(xs),(1:15),xslice);
        lnnan = isnan(data);
        data(lnnan) = eps;
       
        aux = normalize(data(xs,:));
        data(xs,:) = aux;
        D = data(xs,:);
        DATA = [DATA;D];        
    end
    
    %% Data partition (pure vs partial)
    data = [DATA,PL,XS];
    puro_i = find(data(:,19)<4);      
    pv_i = find(data(:,19)>3);
    pv_size = size(pv_i);
    Xref = data(puro_i,(1:15));
    lref = data(puro_i,18);
            
    if rem(pv_size(1),3) == 0
        corte = pv_size(1)/3;
        extra = pv_size(1) - 3*corte;
    else
        corte = floor(pv_size(1)/3);
        extra = pv_size(1) - 3*corte;
    end

    fold1 = 1 : corte;
    fold2 = (max(fold1) + 1) : 2*corte;
    fold3 = (max(fold2) + 1) : 3*corte+extra;
    
    LBLsp2 = zeros(pv_size(1),1);
    LBLsp1p2f = zeros(pv_size(1),3);
    LBLsp1p2 = zeros(pv_size(1),1);
    pv_c = 1;
    MiSom.codebooks = zeros(total_clusters,size(Xref,2),3);
    MiSom.refs = zeros(pv_size(1),2);
    MiSom.lbls = zeros(total_clusters,2);

    clc
    %% Training
    formatSpec = 'Training with sample %u \n';
    fprintf(formatSpec,sample)
 
    for fold = 1 : 3 
        X = [];
        L = [];    
       % Data partition
       switch fold
          case 1
                Xtst = data(pv_i(fold1'),(1:15));  
                ltst = data(pv_i(fold1'),(18));
                stst = data(pv_i(fold1'),(17));
                voxtst = data(pv_i(fold1'),(20));
                Xtrn1 = data(pv_i(fold2'),(1:15));  
                ltrn1 = data(pv_i(fold2'),(18));
                Xtrn2 = data(pv_i(fold3'),(1:15));  
                ltrn2 = data(pv_i(fold3'),(18));
           case 2
                Xtst = data(pv_i(fold2'),(1:15));  
                ltst = data(pv_i(fold2'),(18));
                stst = data(pv_i(fold2'),(17));
                voxtst = data(pv_i(fold2'),(20));
                Xtrn1 = data(pv_i(fold1'),(1:15));  
                ltrn1 = data(pv_i(fold1'),(18));
                Xtrn2 = data(pv_i(fold3'),(1:15));  
                ltrn2 = data(pv_i(fold3'),(18));   
            case 3
                 Xtst = data(pv_i(fold3'),(1:15));  
                ltst = data(pv_i(fold3'),(18));
                stst = data(pv_i(fold3'),(17));
                voxtst = data(pv_i(fold3'),(20));
                Xtrn1 = data(pv_i(fold1'),(1:15));  
                ltrn1 = data(pv_i(fold1'),(18));
                Xtrn2 = data(pv_i(fold2'),(1:15));  
                ltrn2 = data(pv_i(fold2'),(18));                    
           otherwise
               exit;
       end 
       X = [Xref;Xtrn1;Xtrn2];
       L = [lref;ltrn1;ltrn2];
      
       % Adding pseudo labels to the feature descritors
       Data_coded = coding(X,L);

       %% Training
       smI = som_lininit(Data_coded,'msize',mapsize,'lattice','hexa','shape','sheet');
       smC = som_batchtrain(smI,Data_coded,'radius',[3 3],'trainlen',30);
       sm = som_batchtrain(smC,Data_coded,'radius',[1 1],'trainlen',150);
       if fold == 1
           MiSom.SOM_fold1 = sm;
       elseif fold == 2
           MiSom.SOM_fold2 = sm;
       elseif fold == 3
           MiSom.SOM_fold3 = sm;           
       end

       codebook_long = sm.codebook;
       mask_long = sm.mask;
       sm.codebook = sm.codebook(:,(1:15));
       sm.mask = sm.mask(1:15);

        Cc = cell(size(L,1),1);
        for i = 1 : size(L,1)
            Cc{i,1} = num2str(L(i));
        end
        sFrom = som_data_struct(Data_coded(:,(1:15)),'labels',Cc);
        
        sTo = som_autolabel(sm, sFrom,'vote');
        M = sTo.codebook;
        MiSom.codebooks(:,:,fold) = sTo.codebook;
        MiSom.lbls(:,fold) = double(string(sTo.labels));

        % Función para obtener los vecinos adjacentes a distancia 1
        Ne1 = som_unit_neighs(sTo);
        Ne = som_neighborhood(Ne1,2);
        
        L_v = double(string((sTo.labels)));
%         figure(fold)
%         som_cplane(sTo.topol,L_v);
%         title('Prototype clustering')
        nodos = mapsize(1)*mapsize(2);    
        w = ones(3);
        w(5) = 0;
        wnw3 = 3;
        d3=(wnw3-1)/2;
        ai = d3+1; af = row +d3;
        bi = d3+1; bf = col+d3;
        
        %% Testing
        for u = 1 : size(Xtst,1)
            Ci = inf(size(sTo.codebook));
            distXT = inf(tissues,1);

            voxl = voxtst(u);
            sliceg = stst(u);
            
            X1 = Xtst(u,:);
                       
            % Prior knowledge
%             csf = imrotate(atlas_csf.img(:,:,sliceg),90);
%             gm = imrotate(atlas_gm.img(:,:,sliceg),90);
%             wm = imrotate(atlas_wm.img(:,:,sliceg),90);
            umap_ref = find(slix==sliceg);
            csf = csf_map(:,:,umap_ref);
            gm = gm_map(:,:,umap_ref);
            wm = wm_map(:,:,umap_ref);  

            mask = double(imrotate(round(abs(brain_msk.img(:,:,sliceg))),90)); 
                   
            IM=(zeros(row+(2*d3),col+(2*d3))); 
            Icsf=(zeros(row+(2*d3),col+(2*d3))); 
            Igm=(zeros(row+(2*d3),col+(2*d3))); 
            Iwm=(zeros(row+(2*d3),col+(2*d3))); 
            
            IM(ai:af,bi:bf) = mask; 
            Icsf(ai:af,bi:bf) = csf; 
            Igm(ai:af,bi:bf) = gm; 
            Iwm(ai:af,bi:bf) = wm; 
            tempU = zeros(row+(2*d3),col+(2*d3),tissues);
            tempA = zeros(row+(2*d3),col+(2*d3),tissues);
            tempA(:,:,1) = Icsf;
            tempA(:,:,2) = Igm;
            tempA(:,:,3) = Iwm;    
                        
            dist2 = inf(tissues, size(X1, 1));
            dist3 = inf(tissues, size(X1, 1));
            %[row,col] = size(mask);
                                                          
            [bmu,qr] = som_bmus(sTo, X1,2);
            bmu_nei = Ne(bmu,:)< inf;
            Ci(bmu_nei,:)=sTo.codebook(bmu_nei,:);
            distan = zeros(size(Ci, 1), size(1, 1));
            for k = 1:size(Ci, 1)
                distan(k, :) = sqrt(sum(((X1-ones(size(X1,1),1)*Ci(k,:)).^2),2));
            end
            tmp = distan.^(-2/(expo-1));      
            U_new = tmp./(ones(nodos,1)*sum(tmp)); %
            U_new = round(U_new,2);  
            
            for j = 1 : tissues
                c = find(L_v==j);
                cc = distan(c)~=inf;
                ccc = c(cc);
                if isempty(ccc)
%                 distXT(j) = median(distan(ccc));
                else
                    distXT(j) = min(distan(ccc));
                    [xa,ya] = ind2sub([row+(2*d3),col+(2*d3)],voxl);
                    tempU(xa,ya,j) = sum(U_new(ccc));
                end
            end
                                    
            x3 = xa;
            y3 = ya;
            Iu = (tempU(x3-d3:x3+d3,y3-d3:y3+d3,:)); 
            Ia = (tempA(x3-d3:x3+d3,y3-d3:y3+d3,:));
            for q = 1 : tissues
                g = ones(1,3);
                g(q) = 0;
                p1 = sum(sum(Beta*(Iu.*w).^expo));
                p1 = p1(:);
                p1 = g*p1;
                p2 = Gama*sum(sum((Ia.*w).^expo));
                p2 = p2(:);
                p2 = g*p2;   
                dist3(q) = distXT(q) +p1  + p2;
                dist2(q) = distXT(q)  + p2;
            end

            tmp2 = dist2.^(-2/(expo-1));      % calcular nueva partición U
            tmp3 = dist3.^(-2/(expo-1));      % calcular nueva partición U
            U_new2 = tmp2./(ones(tissues , 1)*sum(tmp2)); %
            U_new2 = round(U_new2,2);           
            U_new3 = tmp3./(ones(tissues , 1)*sum(tmp3)); %
            U_new3 = round(U_new3,2);  
            MiSom.refs(pv_c,:) = [bmu,fold];  
            [~,LBLsp2(pv_c)] = max(U_new2);
            [~,LBLsp1p2(pv_c)] = max(U_new3);
            LBLsp1p2f(pv_c,:) = U_new3;
            pv_c = pv_c + 1;
            
%             puase
        end
        
        [qe,te] = som_quality(sTo,Xtst);
        
%         figure
%         som_cplane(sTo.topol,MiSom.lbls(:,count));  
%         title(['SUP qe = ',num2str(qe),' & te = ',num2str(te)])
    end
   
    BMU_map = zeros(length(pv_i),1);
    fold_map = zeros(length(pv_i),1);
    SOM_LBLp2 = zeros(length(data),1);
    SOM_LBLp2(puro_i) = data(puro_i,19);
    SOM_LBLp2(pv_i) = LBLsp2;
    SOM_LBLp1p2 = zeros(length(data),1);
    SOM_LBLp1p2(puro_i) = data(puro_i,19);
    SOM_LBLp1p2(pv_i) = LBLsp1p2;
    SOM_LBLsp1p2f = zeros(length(data),3);
    SOM_LBLsp1p2f(pv_i,:) = LBLsp1p2f;
    
    BMU_map(pv_i) = MiSom.refs(:,1);
    fold_map(pv_i) = MiSom.refs(:,2);
    
    MiSom.SOM_LBLp2 = SOM_LBLp2;
    MiSom.SOM_LBLp1p2 = SOM_LBLp1p2;
    MiSom.SOM_LBLsp1p2f = SOM_LBLsp1p2f;
    MiSom.puro_i = puro_i;  % 17 vs 20       
    MiSom.pv_i = pv_i;
    MiSom.fold_map = fold_map;
    MiSom.BMU_map = BMU_map;
    
    save(['SUPSOM_fuzzy_class_GRDNS2U',num2str(node_row),'x',...
        num2str(node_col),'_3folds_N4itk.mat'],'MiSom')


%%-------------- function
function Data_coded = coding(X,L)
    Data_coded_size = size(X);
    Data_coded = zeros(Data_coded_size(1),18);
    for i = 1 : Data_coded_size(1)
        if L(i) == 1
            Data_coded(i,:) = [X(i,:),0,0,1];
        elseif L(i) == 2
            Data_coded(i,:) = [X(i,:),0,1,0];
        elseif L(i) == 3
            Data_coded(i,:) = [X(i,:),1,0,0];
        end        
    end  
end

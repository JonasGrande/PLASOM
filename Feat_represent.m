%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     Gardens2 algorithm                                                  %
%     Jonás Grande Barreto                                                %
%     María Del Pilar Gómez Gil                                           %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% inputs
%           mri : brain mri volume 
%           brain_msk : binari mask that indicates the whole brain volume

%% output
%           feat_all_ibsr : feature representation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear 
close all
clc
tic
%addpath(('NIfTI_files')) 

rt =[63,201];

% view_nii(mri)
% figure
for subject = 1 : 1% I_N4itk
% mris = load_nii('IBSR_01_ana_strip_NL_N4ITK.nii'); 
mris = load_nii('I_N4itk.nii'); 
brain_msk = load_nii('IBSR_01_ana_brainmask.nii');  
[row,col,dip]=size(mris.img);

am1 = single(round(mris.img));
ay = logical(brain_msk.img);
am2 = imfill(ay,'holes');
am2 = single(am2);
am3 = round(am1.*am2);
gmin = min(am3(:));
gmax = max(am3(:));
am4 = round((am3)*(255/gmax));
Gmin = min(am4(:));
Gmax = max(am4(:));

feat_all_ibsr = zeros(row*col,35,length(rt(subject,1):rt(subject,2)));
LI = zeros(row*col,1,length(rt(subject,1):rt(subject,2)));

% figure
o= 1;
Iwork = [];
for scan = rt(subject,1) : rt(subject,2)
    slix = scan-1:scan+1;
    for p = 1 : length(slix)
        noskull = (abs(am4(:,:,slix(p))));
        noskull = double(round(imrotate(noskull,90)));     
        % ground truth
        if (p ==2) 
            a = abs(mris.img(:,:,slix(p)));
            gt_map = round(imrotate(a,90));
            [row,col] = size(gt_map);
            lbl = zeros(row*col,1);     
            for lb = 1 : 3
                c = find(gt_map == lb);
                lbl(c) = lb; 
            end
        end
        Iwork(:,:,p)=noskull;    
    end
%% Histological references
    wnw3 = 3;
    wnw2 = 5;
    nlevl = 16;
    wnowwav = 16;
    mska = imrotate(am2(:,:,scan),90);
    lina = find(mska);

Iworka = (Iwork);
Iworkb = (single(Iwork));
[row,col]=size(Iworka(:,:,2)); 
d3=(wnw3-1)/2;
d2=(wnw2-1)/2;
offsets = [0 d2; -d2 d2;-d2 0;-d2 -d2];
ai = d3+1; af = row +d3;
bi = d3+1; bf = col+d3;
xi = d2+1; xf = row +d2;
yi = d2+1; yf = col+d2;

Iext=(zeros(row+(2*d3),col+(2*d3),3)); 
Itex = zeros(row+(2*d2),col+(2*d2));

% 3d
for j=1:size(Iworka,3)
     aux=Iworka(:,:,j);
     Iext(ai:af,bi:bf,j) = aux;
end

    %  2d
    Itex(xi:xf,yi:yf) = Iworka(:,:,2);

    %  wavelet
     Iwav = zeros((row+wnowwav-1),(col+wnowwav-1));
     Iwav(1:row,1:col) =Iworka(:,:,2);
     
     allfeat=(zeros(length(lina),35));

    distance=d3; numLevels=nlevl;
    noskull = Iworka(:,:,2);

parfor x = 1 : length(lina) % parallel computing
% for x = 1 : length(lina) % parallel computing
    allfeatv=zeros(1,35);
    [xa,ya] = ind2sub([row,col],lina(x));
    x2 = xa + d2;
    y2 = ya + d2;
    x3 = xa + d3;
    y3 = ya + d3;
    Iwindow = (Iext(x3-d3:x3+d3,y3-d3:y3+d3,:));
    Iww = Iwav(xa:(xa+wnowwav-1),ya:(ya+wnowwav-1));
    
    % Function cooc3d can be found at
    % https://la.mathworks.com/matlabcentral/fileexchange/19058-cooc3d
    % Carl (2021). cooc3d (https://www.mathworks.com/matlabcentral/fileexchange/19058-cooc3d), MATLAB Central File Exchange. Retrieved March 22, 2021. 
    glcm3d = cooc3d(Iwindow,distance,numLevels); 
    glcm3d = (glcm3d);
    
    % Function GLCMFeatures can be found at
    % https://la.mathworks.com/matlabcentral/fileexchange/55034-glcmfeatures-glcm
    % Patrik Brynolfsson (2021). GLCMFeatures(glcm) (https://www.mathworks.com/matlabcentral/fileexchange/55034-glcmfeatures-glcm), MATLAB Central File Exchange. Retrieved March 22, 2021. 
    out = GLCMFeatures(glcm3d);
    [feats]=histfeats(Iwindow);
    Itexa = Itex(x2-d2:x2+d2,y2-d2:y2+d2);
    glcm2d = graycomatrix(Itexa,'Offset',offsets,'Symmetric',true,...
        'NumLevels',nlevl,'GrayLimits',[Gmin,Gmax]);
        stats = graycoprops(glcm2d);
    [c,ss]=wavedec2(Iww,3,'haar');       
    [Ea,Eh,Ev,Ed] = wenergy2(c,ss);       
        % vector
        allfeatv(1) = mean(out.correlation,'omitnan');        
        allfeatv(2) = mean(out.contrast,'omitnan');
        allfeatv(3) = mean(out.energy,'omitnan');
        allfeatv(4) = mean(out.homogeneity,'omitnan');
        allfeatv(5) = mean(out.entropy,'omitnan');
        allfeatv(6) = mean(out.sumOfSquaresVariance,'omitnan');
        allfeatv(7) = mean(out.sumAverage,'omitnan');
        allfeatv(8) = mean(out.dissimilarity,'omitnan');
        allfeatv(9) = mean(out.clusterShade,'omitnan');
        allfeatv(10) = mean(out.clusterProminence,'omitnan');
        allfeatv(11) = mean(out.maximumProbability,'omitnan');
        allfeatv(12) = mean(out.differenceVariance,'omitnan');
        allfeatv(13)=Iworka(xa,ya,2); 
        allfeatv(14)=mean(Iwindow(:)); 
        allfeatv(15)=var(Iwindow(:)); 
        allfeatv(16)=feats.Mean; 
        allfeatv(17)=feats.Variance; 
        allfeatv(18)=feats.Entropy; 
        allfeatv(19)=feats.Energy; 
        allfeatv(20)=feats.Skewness; 
        allfeatv(21)=feats.Kurtosis; 
        allfeatv(22) =mean(stats.Contrast,'omitnan');
        allfeatv(23) =mean(stats.Correlation,'omitnan');
        allfeatv(24) =mean(stats.Energy,'omitnan');
        allfeatv(25) =mean(stats.Homogeneity,'omitnan');
        allfeatv(26) = Ea;
        allfeatv(27) = Eh(1);
        allfeatv(28) = Eh(2);
        allfeatv(29) = Eh(3);
        allfeatv(30) = Ev(1);
        allfeatv(31) = Ev(2);
        allfeatv(32) = Ev(3);
        allfeatv(33) = Ed(1);
        allfeatv(34) = Ed(2);
        allfeatv(35) = Ed(3);
        oo = 1;

        allfeat(x,:) = allfeatv;
end
data = zeros(row*col,35); 
data(lina,:)=allfeat;

feat_all_ibsr(1:length(lina),:,o) = allfeat;
oi = (length(lina)); 
LI((1:oi),:,o) =lina; 

% BI = reshape(data(:,13),row,col);    
% imshow(BI,[],'InitialMagnification','fit')
% pause(0.01)
% imshow(imags,[],'InitialMagnification','fit')


formatSpec = 'Subject..%u...scan %u/%u ... Progress %3.3f%% \r';
A1 = (o/length(rt(subject,1):rt(subject,2)))*100; 
fprintf(formatSpec,subject,o,length(rt(subject,1):rt(subject,2)),A1)

o = o+1;
end
    save('feats_Subject_x_N4itk','feat_all_ibsr')
% toc
end
toc

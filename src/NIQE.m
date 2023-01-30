path= "D:\Test\*.png";
imagefiles = dir([path]);
nfiles = length(imagefiles);   % Number of files found
result = 0.0
for ii=1:nfiles
   currentfilename = imagefiles(ii).name;
   currentfolder = imagefiles(ii).folder;
   chr = '\';
   path_to_file = append(currentfolder,chr,currentfilename);
   disp(path_to_file)
   currentimage = imread(path_to_file);
   niqe_current = niqe(currentimage)
   result = result + niqe_current
   fprintf('NIQE score for original image is %0.4f.\n',niqe_current)
end
fprintf('NIQE score for all image is %0.4f.\n',result);
fprintf('Average NIQE score for original image is %0.4f.\n',result/nfiles);
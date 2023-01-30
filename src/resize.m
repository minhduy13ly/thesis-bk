path= "C:\Users\Admin\Desktop\set5\Urban100\*.png";
des_path = "C:\Users\Admin\Desktop\set5\Urban100_LR\"
imagefiles = dir([path]);
nfiles = length(imagefiles);
for ii=1:nfiles
   currentfilename = imagefiles(ii).name;
   currentfolder = imagefiles(ii).folder;
   chr = '\';
   path_to_file = append(currentfolder,chr,currentfilename);
   disp(path_to_file)
   currentimage = imread(path_to_file);
   resize_img = imresize(currentimage, 0.25, 'bicubic');
   imwrite(resize_img,des_path  + currentfilename)
   
end
   
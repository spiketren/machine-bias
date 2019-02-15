%pkg load image % loading octave-image
DIR='./face128_test/';
files=dir([DIR './*.jpg']);

for i=1:length(files)
    %disp(files(i).name);
    img=rgb2gray(imread([DIR files(i).name]));
    img2=img;
    idx_eye=0;
    while(abs(idx_eye-50)>20)
       [val,idx_eye]=min(sum(img2(:,32:96),2));
       img2(idx_eye,:)=255;
    end
    img2=img;
    idx_nose=0;
    while(abs(idx_nose-70)>10)
       [val,idx_nose]=min(sum(img2(:,32:96),2));
       img2(idx_nose,:)=255;
    end
    img2=img;
    idx_axis=0;
    while(abs(idx_axis-64)>15)
      [val,idx_axis]=max(sum(img2(idx_eye:idx_nose,:)));
      img2(:,idx_axis)=0;
    end
    
    %Visualize the axis of symmetry:------
    img(:,idx_axis)=0;
    imshow(img);
    %-------------------------------------
    
    d=round(min(128-idx_axis,idx_axis-1)/2);
    if(idx_axis>64)
      dif=abs(img(32:96,(idx_axis-d):(idx_axis-1))-img(32:96,(idx_axis+1):(idx_axis+d)));
    else
      dif=abs(img(32:96,(idx_axis-d):(idx_axis-1))-img(32:96,(idx_axis+1):(idx_axis+d)));
    end
    c=[]; %correlations
    for i=1:d
      c=[c corr(double(img(32:96,idx_axis-i)),double(img(32:96,idx_axis+i)))];
    end
    disp(sprintf('%f',mean(c)))
end

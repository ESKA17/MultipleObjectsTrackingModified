I = imread('red.png');
figure (1)
Isub = imsubtract(I(:,:,1), rgb2gray(I));
Isub = medfilt2(Isub, [3 3]);

Isub = imbinarize(Isub,0.1);

Isub = bwareaopen(Isub, 300);
Isub = uint8(repmat(Isub, 1, 1, 3));
Isub(Isub == 0) = 255;
Isub(Isub(:, :, 1) == 1) = 255;
imshowpair(I, Isub, 'montage')






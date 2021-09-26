% Mask a leaf out of an RGB image.
clc;    % Clear the command window.
clear all;
close all;
workspace;  % Make sure the workspace panel is showing.
format long g;
format compact;
fontSize = 24;

%===============================================================================
% Read in leaf color demo image.
folder = pwd
baseFileName = 'leaf.jpg';
% Get the full filename, with path prepended.
fullFileName = fullfile(folder, baseFileName);
if ~exist(fullFileName, 'file')
	% Didn't find it there.  Check the search path for it.
	fullFileName = baseFileName; % No path this time.
	if ~exist(fullFileName, 'file')
		% Still didn't find it.  Alert user.
		errorMessage = sprintf('Error: %s does not exist.', fullFileName);
		uiwait(warndlg(errorMessage));
		return;
	end
end
rgbImage = imread(fullFileName);
% Get the dimensions of the image.  numberOfColorBands should be = 3.
[rows, columns, numberOfColorChannels] = size(rgbImage);
% Display the original color image.
subplot(2, 2, 1);
imshow(rgbImage);
title('Original Color Image', 'FontSize', fontSize, 'Interpreter', 'None');
% Enlarge figure to full screen.
set(gcf, 'Units', 'Normalized', 'Outerposition', [0, 0, 1, 1]);

% Extract the individual red, green, and blue color channels.
% redChannel = rgbImage(:, :, 1);
% greenChannel = rgbImage(:, :, 2);
blueChannel = rgbImage(:, :, 3);

% Create a mask of the background only.
mask = blueChannel > 200;
% Display the mask image.
subplot(2, 2, 2);
imshow(mask);
title('Mask Image', 'FontSize', fontSize, 'Interpreter', 'None');

% Mask out the leaf, leaving only the background.
% Mask the image using bsxfun() function
maskedRgbImage = bsxfun(@times, rgbImage, cast(mask, 'like', rgbImage));
% Display the mask image.
subplot(2, 2, 3);
imshow(maskedRgbImage);
title('Background-Only Image', 'FontSize', fontSize, 'Interpreter', 'None');

% Mask out the background, leaving only the leaf.
% Mask the image using bsxfun() function
maskedRgbImage = bsxfun(@times, rgbImage, cast(~mask, 'like', rgbImage));
% Display the mask image.
subplot(2, 2, 4);
imshow(maskedRgbImage);
title('Leaf-Only Image', 'FontSize', fontSize, 'Interpreter', 'None');
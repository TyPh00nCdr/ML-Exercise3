function gda
  ## Warning: Immediate in-memory operation (not lazy)
  positives = cellfun(@imreadtodouble, glob("../PythonSrc/data/positives/p*.png"), "UniformOutput", false);
  negatives = cellfun(@imreadtodouble, glob("../PythonSrc/data/negatives/n*.png"), "UniformOutput", false);
  
  uPos = mean(features(positives));
  uNeg = mean(features(negatives));
 
  c = 0;
  feats = [features(positives); features(negatives)]';
  for feat = feats;
    c += feat * feat';
  end
  c /= rows(feats);
  
  # cov = arrayfun(@(feat) inspect(feat), feats(:), "UniformOutput", false);
  
endfunction

function i = inspect (x)
  j = x;
  i = j;
endfunction

function img = imreadtodouble (im)
  # Load a single image from path (im) as a matrix of doubles  
  img = double(imread(im));
endfunction

function feats = features (doubleimgs)
  # Collect all features from the cell array of (24 x 24 x 3) image matrices
  feats = cell2mat(cellfun("rgbgrayfeats", doubleimgs, "UniformOutput", false));
  
  function allfeats = rgbgrayfeats (doubleimg)
    # Collect all features from a single (24 x 24 x 3) image matrix
    allfeats = [feature(doubleimg(:, :, 1)) feature(doubleimg(:, :, 2)) feature(doubleimg(:, :, 3)) feature(rgb2gray(doubleimg))];
  endfunction
endfunction

function feat = feature (doubleimg)
  # Collect all features from a single (24 x 24) image matrix
  feat = [max(doubleimg(:)), min(doubleimg(:)), mean(doubleimg(:)), std(doubleimg(:)), var(doubleimg(:))];
endfunction

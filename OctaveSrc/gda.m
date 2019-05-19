function gda
  ## Warning: Immediate in-memory operation (not lazy)
  positives = cellfun(@imreadtodouble, glob("../PythonSrc/data/positives/p*.png"), "UniformOutput", false);
  negatives = cellfun(@imreadtodouble, glob("../PythonSrc/data/negatives/n*.png"), "UniformOutput", false);
  
  feats_pos = features(positives); 
  feats_neg = features(negatives);
  
  uPos = mean(feats_pos);
  uNeg = mean(feats_neg);
 
  c = 0;
  for feat = feats_pos';
    c += (feat - uPos) * (feat - uPos)';
  end
  for feat = feats_neg';
    c += (feat - uNeg) * (feat - uNeg)';
  end
  c /= (rows(feats_pos) + rows(feats_neg));
  
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

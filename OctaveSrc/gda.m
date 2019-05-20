function gda
  ## Warning: Immediate in-memory operation (not lazy)
  positives = cellfun(@imreadtodouble, glob("../PythonSrc/data/positives/p*.png"), "UniformOutput", false);
  negatives = cellfun(@imreadtodouble, glob("../PythonSrc/data/negatives/n*.png"), "UniformOutput", false);
  
  feats_pos = features(positives); 
  feats_neg = features(negatives);
  
  uPos = mean(feats_pos)';
  uNeg = mean(feats_neg)';
 
  global covSigma = 0;
  for feat = feats_pos';
    covSigma += (feat - uPos) * (feat - uPos)';
  end
  for feat = feats_neg';
    covSigma += (feat - uNeg) * (feat - uNeg)';
  end
  covSigma /= (rows(feats_pos) + rows(feats_neg));
  global invCovSigma = inv(covSigma);
  global detCovSigma = det(covSigma);
  
  disp(detCovSigma);
  
endfunction

function prob = probgiveny (x, y)
  leftTerm = 1 / ((2 * pi)^(length(feat) / 2) * sqrt(detCovSigma));
  if (y == 0)
    prob = leftTerm * exp(-0.5 * (x - uNeg)' * invCovSigma * (x - uNeg));
  elseif (y == 1) 
    prob = leftTerm * exp(-0.5 * (x - uPos)' * invCovSigma * (x - uPos));
  endif
endfunction

function img = imreadtodouble (im)
  ## Load a single image from path (im) as a matrix of doubles  
  img = double(imread(im));
endfunction

function feats = features (doubleimgs)
  ## Collect all features from the cell array of (24 x 24 x 3) image matrices
  feats = cell2mat(cellfun("rgbfeats", doubleimgs, "UniformOutput", false));
  
  function allfeats = rgbfeats (doubleimg)
    ## Collect all features from a single (24 x 24 x 3) image matrix
    allfeats = [feature(doubleimg(:, :, 1)) feature(doubleimg(:, :, 2)) feature(doubleimg(:, :, 3))] #feature(rgb2gray(doubleimg))];
    # allfeats = feature(rgb2gray(doubleimg));
  endfunction
endfunction

function feat = feature (doubleimg)
  ## Collect all features from a single (24 x 24) image matrix
  feat = [max(doubleimg(:)), min(doubleimg(:)), mean(doubleimg(:)), std(doubleimg(:)), var(doubleimg(:))];
endfunction

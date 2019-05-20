function gda ()
  # Clear all previously persisted global and persistent variables.
  clear all;
  # Warning: Immediate in-memory operation (not lazy)
  positives = cellfun(@imreadtodouble, glob("../PythonSrc/data/positives/p*.png"), "UniformOutput", false);
  negatives = cellfun(@imreadtodouble, glob("../PythonSrc/data/negatives/n*.png"), "UniformOutput", false);
  
  featsPos = features(positives); 
  featsNeg = features(negatives);
  
  global uPos = mean(featsPos)';
  global uNeg = mean(featsNeg)';
 
  covSigma = zeros(columns(featsPos), rows(uPos));
  for feat = featsPos';
    covSigma += (feat - uPos) * (feat - uPos)';
  end
  for feat = featsNeg';
    covSigma += (feat - uNeg) * (feat - uNeg)';
  end
  covSigma /= (rows(featsPos) + rows(featsNeg));
  global invCovSigma = inv(covSigma);
  global detCovSigma = det(covSigma);
  
  ## Output:
  printf("Sigma positive semi-definite?: %s\n", mat2str(all(eigs(covSigma) >= 0)));
  printf("Sigma's determinant: %d\n", detCovSigma);
  
  printf("----------------------\n");
  
  printf("Label: 1, with prob: 1\n");
  for feat = featsPos';
    [prob, label] = classify(feat);
    printf("Label: %d, with prob: %d\n", label, prob);
  end
  
  printf("----------------------\n");
  
  printf("Negative Images:\n");
  for feat = featsNeg';
    [prob, label] = classify(feat);
    printf("Label: %d, with prob: %d\n", label, prob);
  end
endfunction

function [prob, label] = classify (x)
  probX = probgiveny(x, 0) * 0.5 + probgiveny(x, 1) * 0.5;
  [prob, label] = max([((probgiveny(x, 0) * 0.5) / probX) ((probgiveny(x, 1) * 0.5) / probX)]);
  # MatLab/Octave starts idx at 1, not 0...
  label -= 1;
endfunction

function prob = probgiveny (x, y)
  ## Probability of feature vector x given label y: p(x | y = {0, 1})
  global invCovSigma detCovSigma uPos uNeg;
  persistent leftTerm;
  if isempty(leftTerm)
    leftTerm = 1 / ((2 * pi)^(length(x) / 2) * sqrt(detCovSigma));
  endif
  if y == 0
    prob = leftTerm * exp(-0.5 * (x - uNeg)' * invCovSigma * (x - uNeg));
  elseif y == 1
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
    allfeats = [feature(doubleimg(:, :, 1)) feature(doubleimg(:, :, 2)) feature(doubleimg(:, :, 3))]; # feature(rgb2gray(doubleimg))];
    # allfeats = feature(rgb2gray(doubleimg));
  endfunction
endfunction

function feat = feature (doubleimg)
  ## Collect all features from a single (24 x 24) image matrix
  feat = [max(doubleimg(:)), min(doubleimg(:)), mean(doubleimg(:)), std(doubleimg(:)), var(doubleimg(:))];
endfunction

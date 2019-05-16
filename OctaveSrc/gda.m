function gda
  ## Warning: Immediate in-memory operation (not lazy)
  positives = cellfun(@imread, glob("../PythonSrc/data/positives/p*.png"), "UniformOutput", false);
  negatives = cellfun(@imread, glob("../PythonSrc/data/negatives/n*.png"), "UniformOutput", false);
 
  ## figure, imshow(positives{1});
endfunction
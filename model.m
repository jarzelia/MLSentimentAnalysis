
%% Initialization
clear ; close all; clc

% Load training data (created by 'parser.py')
path_theData = 'dataio/theData.dat';

if(exist(path_theData))
   [test_class, test_tf, test_words] = textread(path_theData, '%d %d %s');
else
	% Error: DNE
   disp(['Could not open ' path_theData '. File does not exist.']);
end


%% ============ Part 1: Compute Features Cost and Gradient ============

% Computing features f1, f2, ..., fn

% Some random test
someFile = fopen('test/0.txt');
fclose(someFile);
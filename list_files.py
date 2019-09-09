import os
images = sorted(os.listdir('data/test/frames_c/'))
fp = open('test_path.csv', 'w')
for i, im in enumerate(images):
    fp.write(str(i)+', '+ 'data/test/frames_c/' + im + '\n')

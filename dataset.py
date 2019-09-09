from torch.utils.data import Dataset
class imageDataset(Dataset):
    def __init__(self, datatype):
        self.labels = [float(l) for l in open('data/' + datatype + '.txt').readlines()]
        #print(type(self.labels[-1]))
        self.create_f2p(open(datatype + '_path.csv'))
        self.paths = self.get_paths()
        #print(self.paths)

    def get_paths(self):
        paths = []
        for i in range(len(self.f2p)):
            paths.append(self.f2p[str(i)])
        return paths

    def __len__(self):
        return len(self.paths)-8+1

    def __getitem__(self, idx):
        #for i in range(10):
            #print(self.paths[idx+i])
        return self.labels[idx+7], self.paths[idx:idx+8]

    def create_f2p(self, fp):
        self.f2p=dict()
        for row in fp:
            txt = row.split(',')
            k,v=txt[0], txt[1].split('\n')[0].strip()
            self.f2p[k]=v
        return

import os
from abc import abstractmethod

import torch
from torch.utils.data import Dataset
from utils import makedir_exist_ok, save, load
from .utils import download_url, extract_file


class Vocab:
    def __init__(self):
        self.symbol_to_index = {}
        self.index_to_symbol = []
        self.symbol_counts = {}

    def add(self, symbol):
        if symbol not in self.symbol_to_index:
            self.index_to_symbol.append(symbol)
            self.symbol_to_index[symbol] = len(self.index_to_symbol) - 1
        return

    def delete(self, symbol):
        if symbol in self.symbol_to_index:
            self.index_to_symbol.remove(symbol)
            self.symbol_to_index.pop(symbol, None)
        return

    def count(self, input):
        if isinstance(input, int):
            count = self.symbol_counts[self.index_to_symbol[input]]
        elif isinstance(input, str):
            count = self.symbol_counts[input]
        else:
            raise ValueError('wrong input data type')
        return count

    def symbol_counts(self):
        counts = sorted(self.symbol_counts.items(), key=lambda x: x[1], reverse=True)
        return counts

    def __len__(self):
        return len(self.index_to_symbol)

    def __getitem__(self, input):
        if isinstance(input, int):
            output = self.index_to_symbol[input]
        elif isinstance(input, str):
            output = self.symbol_to_index[input]
        else:
            raise ValueError('wrong input data type')
        return output

    def __contains__(self, input):
        if isinstance(input, int):
            exist = input in self.index_to_symbol
        elif isinstance(input, str):
            exist = input in self.symbol_to_index
        else:
            raise ValueError('wrong input data type')
        return exist


class LanguageModeling(Dataset):
    def __init__(self, root, split, download):
        self.data_name = None
        self.root = os.path.expanduser(root)
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')
        self.data = load(os.path.join(self.processed_folder, '{}.pt'.format(split)))
        self.vocab = load(os.path.join(self.processed_folder, 'meta.pt'.format(split)))

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    def _check_exists(self):
        return os.path.exists(self.processed_folder)

    @abstractmethod
    def download(self):
        raise NotImplementedError

    def __repr__(self):
        fmt_str = 'Dataset ' + self.data_name + '\n'
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str


class PennTreebank(LanguageModeling):
    data_name = 'PennTreebank'
    urls = ['https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt',
           'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt',
           'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt']
    data_file = {'train':'ptb.train.txt', 'validation':'ptb.valid.txt', 'test':'ptb.test.txt'}

    def __init__(self, root, split, download=False):
        super(PennTreebank, self).__init__(root, split, download)

    def download(self):
        if self._check_exists():
            return
        makedir_exist_ok(os.path.join(self.raw_folder))
        for url in self.urls:
            filename = os.path.basename(url)
            download_url(url, root=self.raw_folder, filename=filename, md5=None)
        vocab = Vocab()
        for split in self.data_file:
            token_path = os.path.join(self.raw_folder, self.data_file[split])
            num_tokens = 0
            with open(token_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.split() + [u'<eos>']
                    num_tokens += len(line)
                    for symbol in line:
                        vocab.add(symbol)
            with open(token_path, 'r', encoding='utf-8') as f:
                data = torch.LongTensor(num_tokens)
                i = 0
                for line in f:
                    line = line.split() + [u'<eos>']
                    for symbol in line:
                        data[i] = vocab.symbol_to_index[symbol]
                        i += 1
            save(data, os.path.join(self.processed_folder, '{}.pt'.format(split)))
        save(vocab, os.path.join(self.processed_folder, 'meta.pt'))
        return


class WikiText2(LanguageModeling):
    data_name = 'WikiText2'
    url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'
    data_file = {'train':'wiki.train.tokens', 'validation':'wiki.valid.tokens', 'test':'wiki.test.tokens'}

    def __init__(self, root, split, download=False):
        super(WikiText2, self).__init__(root, split, download)

    def download(self):
        if self._check_exists():
            return
        makedir_exist_ok(os.path.join(self.raw_folder))
        filename = os.path.basename(self.url)
        file_path = os.path.join(self.raw_folder, filename)
        download_url(self.url, root=self.raw_folder, filename=filename, md5=None)
        extract_file(file_path)
        vocab = Vocab()
        for split in self.data_file:
            token_path = os.path.join(self.raw_folder, 'wikitext-2', self.data_file[split])
            num_tokens = 0
            with open(token_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.split() + [u'<eos>']
                    num_tokens += len(line)
                    for symbol in line:
                        vocab.add(symbol)
            with open(token_path, 'r', encoding='utf-8') as f:
                data = torch.LongTensor(num_tokens)
                i = 0
                for line in f:
                    line = line.split() + [u'<eos>']
                    for symbol in line:
                        data[i] = vocab.symbol_to_index[symbol]
                        i += 1
            save(data, os.path.join(self.processed_folder, '{}.pt'.format(split)))
        save(vocab, os.path.join(self.processed_folder, 'meta.pt'))
        return


class WikiText103(LanguageModeling):
    data_name = 'WikiText103'
    url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip'
    data_file = {'train':'wiki.train.tokens', 'validation':'wiki.valid.tokens', 'test':'wiki.test.tokens'}

    def __init__(self, root, split, download=False):
        super(WikiText103, self).__init__(root, split, download)

    @property
    def data_path(self):
        return os.path.join(self.root, 'wiki')

    def download(self):
        if self._check_exists():
            return
        makedir_exist_ok(os.path.join(self.raw_folder))
        filename = os.path.basename(self.url)
        file_path = os.path.join(self.raw_folder, filename)
        download_url(self.url, root=self.raw_folder, filename=filename, md5=None)
        extract_file(file_path)
        vocab = Vocab()
        for split in self.data_file:
            token_path = os.path.join(self.raw_folder, 'wikitext-103', self.data_file[split])
            num_tokens = 0
            with open(token_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.split() + [u'<eos>']
                    num_tokens += len(line)
                    for symbol in line:
                        vocab.add(symbol)
            with open(token_path, 'r', encoding='utf-8') as f:
                data = torch.LongTensor(num_tokens)
                i = 0
                for line in f:
                    line = line.split() + [u'<eos>']
                    for symbol in line:
                        data[i] = vocab.symbol_to_index[symbol]
                        i += 1
            save(data, os.path.join(self.processed_folder, '{}.pt'.format(split)))
        save(vocab, os.path.join(self.processed_folder, 'meta.pt'))
        return

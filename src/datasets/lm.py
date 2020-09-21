import os
import torch
from abc import abstractmethod
from torch.utils.data import Dataset
from utils import check_exists, makedir_exist_ok, save, load
from .utils import download_url, extract_file


class Vocab:
    def __init__(self):
        self.symbol_to_index = {u'<ukn>': 0, u'<eos>': 1}
        self.index_to_symbol = [u'<ukn>', u'<eos>']

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

    def __len__(self):
        return len(self.index_to_symbol)

    def __getitem__(self, input):
        if isinstance(input, int):
            if len(self.index_to_symbol) > input >= 0:
                output = self.index_to_symbol[input]
            else:
                output = u'<ukn>'
        elif isinstance(input, str):
            if input not in self.symbol_to_index:
                output = self.symbol_to_index[u'<ukn>']
            else:
                output = self.symbol_to_index[input]
        else:
            raise ValueError('Not valid data type')
        return output

    def __contains__(self, input):
        if isinstance(input, int):
            exist = len(self.index_to_symbol) > input >= 0
        elif isinstance(input, str):
            exist = input in self.symbol_to_index
        else:
            raise ValueError('Not valid data type')
        return exist


class LanguageModeling(Dataset):
    def __init__(self, root, split):
        self.root = os.path.expanduser(root)
        self.split = split
        if not check_exists(self.processed_folder):
            self.process()
        self.token = load(os.path.join(self.processed_folder, '{}.pt'.format(split)))
        self.vocab = load(os.path.join(self.processed_folder, 'meta.pt'.format(split)))

    def __getitem__(self, index):
        input = {'label': self.token[index]}
        return input

    def __len__(self):
        return len(self.token)

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    def _check_exists(self):
        return os.path.exists(self.processed_folder)

    @abstractmethod
    def process(self):
        raise NotImplementedError

    @abstractmethod
    def download(self):
        raise NotImplementedError

    def __repr__(self):
        fmt_str = 'Dataset {}\nRoot: {}\nSplit: {}'.format(
            self.__class__.__name__, self.root, self.split)
        return fmt_str


class PennTreebank(LanguageModeling):
    data_name = 'PennTreebank'
    file = [('https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt', None),
            ('https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt', None),
            ('https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt', None)]

    def __init__(self, root, split):
        super().__init__(root, split)

    def process(self):
        if not check_exists(self.raw_folder):
            self.download()
        train_set, valid_set, test_set, meta = self.make_data()
        save(train_set, os.path.join(self.processed_folder, 'train.pt'))
        save(valid_set, os.path.join(self.processed_folder, 'valid.pt'))
        save(test_set, os.path.join(self.processed_folder, 'test.pt'))
        save(meta, os.path.join(self.processed_folder, 'meta.pt'))
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        for (url, md5) in self.file:
            filename = os.path.basename(url)
            download_url(url, self.raw_folder, filename, md5)
            extract_file(os.path.join(self.raw_folder, filename))
        return

    def make_data(self):
        vocab = Vocab()
        read_token(vocab, os.path.join(self.raw_folder, 'ptb.train.txt'))
        read_token(vocab, os.path.join(self.raw_folder, 'ptb.valid.txt'))
        train_token = make_token(vocab, os.path.join(self.raw_folder, 'ptb.train.txt'))
        valid_token = make_token(vocab, os.path.join(self.raw_folder, 'ptb.valid.txt'))
        test_token = make_token(vocab, os.path.join(self.raw_folder, 'ptb.test.txt'))
        return train_token, valid_token, test_token, vocab


class WikiText2(LanguageModeling):
    data_name = 'WikiText2'
    file = [('https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip', None)]

    def __init__(self, root, split):
        super().__init__(root, split)

    def process(self):
        if not check_exists(self.raw_folder):
            self.download()
        train_set, valid_set, test_set, meta = self.make_data()
        save(train_set, os.path.join(self.processed_folder, 'train.pt'))
        save(valid_set, os.path.join(self.processed_folder, 'valid.pt'))
        save(test_set, os.path.join(self.processed_folder, 'test.pt'))
        save(meta, os.path.join(self.processed_folder, 'meta.pt'))
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        for (url, md5) in self.file:
            filename = os.path.basename(url)
            download_url(url, self.raw_folder, filename, md5)
            extract_file(os.path.join(self.raw_folder, filename))
        return

    def make_data(self):
        vocab = Vocab()
        read_token(vocab, os.path.join(self.raw_folder, 'wikitext-2', 'wiki.train.tokens'))
        read_token(vocab, os.path.join(self.raw_folder, 'wikitext-2', 'wiki.train.tokens'))
        train_token = make_token(vocab, os.path.join(self.raw_folder, 'wikitext-2', 'wiki.train.tokens'))
        valid_token = make_token(vocab, os.path.join(self.raw_folder, 'wikitext-2', 'wiki.valid.tokens'))
        test_token = make_token(vocab, os.path.join(self.raw_folder, 'wikitext-2', 'wiki.test.tokens'))
        return train_token, valid_token, test_token, vocab


class WikiText103(LanguageModeling):
    data_name = 'WikiText103'
    file = [('https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip', None)]

    def __init__(self, root, split):
        super().__init__(root, split)

    def process(self):
        if not check_exists(self.raw_folder):
            self.download()
        train_set, valid_set, test_set, meta = self.make_data()
        save(train_set, os.path.join(self.processed_folder, 'train.pt'))
        save(valid_set, os.path.join(self.processed_folder, 'valid.pt'))
        save(test_set, os.path.join(self.processed_folder, 'test.pt'))
        save(meta, os.path.join(self.processed_folder, 'meta.pt'))
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        for (url, md5) in self.file:
            filename = os.path.basename(url)
            download_url(url, self.raw_folder, filename, md5)
            extract_file(os.path.join(self.raw_folder, filename))
        return

    def make_data(self):
        vocab = Vocab()
        read_token(vocab, os.path.join(self.raw_folder, 'wikitext-103', 'wiki.train.tokens'))
        read_token(vocab, os.path.join(self.raw_folder, 'wikitext-103', 'wiki.train.tokens'))
        train_token = make_token(vocab, os.path.join(self.raw_folder, 'wikitext-103', 'wiki.train.tokens'))
        valid_token = make_token(vocab, os.path.join(self.raw_folder, 'wikitext-103', 'wiki.valid.tokens'))
        test_token = make_token(vocab, os.path.join(self.raw_folder, 'wikitext-103', 'wiki.test.tokens'))
        return train_token, valid_token, test_token, vocab


def read_token(vocab, token_path):
    with open(token_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.split() + [u'<eos>']
            for symbol in line:
                vocab.add(symbol)
    return


def make_token(vocab, token_path):
    token = []
    with open(token_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.split() + [u'<eos>']
            for symbol in line:
                token.append(vocab[symbol])
    token = torch.tensor(token, dtype=torch.long)
    return token
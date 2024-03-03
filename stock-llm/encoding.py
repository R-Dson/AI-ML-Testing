import torch
from torch.utils.data import Dataset, DataLoader

class FinancialDataset(Dataset):
    def __init__(self, data):
        self.data = []
        for item in data:
            l = list(item.get("input").values())
            list_items = []
            for input in l:
                if isinstance(input, list):
                    list_items.extend(input)
                else:
                    list_items.append(input)

            self.data.append({'ratings': item.get('ratings'), 'input': list_items})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        for i in range(len(item['input'])):
            if item['input'][i] is None:
                item['input'][i] = torch.tensor(0)
        tensor_item = self.item_to_tensor(item)
        
        features = tensor_item['input']
        label = tensor_item['ratings']
        return features, label

    def item_to_tensor(self, item):
        input = item.get('input')
        ratings = item.get('ratings')
        
        if isinstance(input, list):
            input = torch.Tensor(input)
        else:
            input = torch.tensor(input)

        if isinstance(ratings, list):
            ratings = torch.Tensor(ratings)
        else:
            ratings = torch.tensor(ratings)
        return {'ratings': ratings, 'input': input}

def flatten_and_pad(data):
    # Flatten and pad the data
    flattened_data = []
    flatten_data_set = {}
    max_length = 0
    for item in data:
        for ticker, ticker_data in item.items():
            flattened_ticker_data = {}
            for _, sub_data in ticker_data.items():
                if isinstance(sub_data, list):
                    for sub_item in sub_data:
                        flattened_sub_data = {}
                        for key, value in sub_item.items():
                            if isinstance(value, dict):
                                flattened_sub_data.update(value)
                            else:
                                flattened_sub_data[key] = value
                        flattened_ticker_data.update(flattened_sub_data)
                        
                elif isinstance(sub_data, dict):
                    flattened_ticker_data.update(sub_data)
            
            for k, v in flattened_ticker_data.items():
                if isinstance(v, list):
                    max_length = max(max_length, len(v))
                    flattened_ticker_data[k] = [ x if x != -1 else 0 for x in v ]
                else:
                    x = flattened_ticker_data[k]
                    flattened_ticker_data[k] = x if x != -1 else 0

            flatten_data_set[ticker] = flattened_ticker_data
            flattened_data.append(flattened_ticker_data)

    all_keys_lists = set()
    all_keys_items = set()
    for item in flattened_data:
        keys = item.keys()
        for key in keys:
            if isinstance(item[key], list):
                all_keys_lists.add(key)
            else:
                all_keys_items.add(key)
        max_length = max(len(v) if isinstance(v, list) else 0 for v in item.values())

    for item in flattened_data:
        missing_keys_lists = all_keys_lists.difference(item.keys()) 
        for key in missing_keys_lists:
            k_v = [0] * max_length
            item[key] = k_v

        missing_keys_items = all_keys_items.difference(item.keys()) 
        for key in missing_keys_items:
            k_v = 0
            item[key] = k_v

        # Ensure all dictionaries have the same length
        for k, v in item.items():
            if isinstance(v, list):
                pad_size = max_length - len(v)
                item[k] = v + [0] * pad_size

    return flatten_data_set

def preprocess_data(data):
    data_set = flatten_and_pad(data)
    
    li = []
    for i in (data):
        k, v = i.popitem()
        v['input'] = data_set[k]
        li.append(v)

    return FinancialDataset(li)

def main():
    import json
    batch_size = 64
    path = 'stocks_data.json'
    f = open(path)
    data = json.load(f)
    
    processed_data = preprocess_data(data)
    for i in processed_data:
        if any(i[0]) is None:
            pass
        print(len(i[0]))
    dataloader = DataLoader(processed_data, batch_size=batch_size, shuffle=True)

    pass

if __name__ == '__main__':
    main()

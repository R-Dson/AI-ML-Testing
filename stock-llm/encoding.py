import torch
from torch.utils.data import Dataset, DataLoader

### TODO...

class FinancialDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoded_item = self.encode_item(item)
        return encoded_item

    def encode_item(self, item):
        encoded_item = {}

        # Encode income statement data
        income_statement = item.get("financial_table_incomeStatement", {})
        encoded_income_statement = self.encode_table(income_statement)
        encoded_item["income_statement"] = encoded_income_statement

        # Encode balance sheet data
        balance_sheet = item.get("financial_table_balanceSheet", {})
        encoded_balance_sheet = self.encode_table(balance_sheet)
        encoded_item["balance_sheet"] = encoded_balance_sheet

        # Encode cash flow statement data
        cash_flow_statement = item.get("financial_table_cashFlowStatement", {})
        encoded_cash_flow_statement = self.encode_table(cash_flow_statement)
        encoded_item["cash_flow_statement"] = encoded_cash_flow_statement

        return encoded_item

    def encode_table(self, table):
        encoded_table = {}
        for key, values in table.items():
            encoded_values = [torch.tensor(value) for value in values if value != []]
            if encoded_values:
                max_length = max(len(tensor) for tensor in encoded_values)
                padded_values = [torch.nn.functional.pad(tensor, (0, max_length - len(tensor))) for tensor in encoded_values]
                encoded_table[key] = torch.stack(padded_values)
        return encoded_table

def preprocess_data(data):
    dataset = FinancialDataset(data)
    return dataset

def main():
    import json
    batch_size = 5
    path = 'stocks_data.json'
    f = open(path)
    data = json.load(f)
    processed_data = preprocess_data(data)
    dataloader = DataLoader(processed_data, batch_size=batch_size, shuffle=True)

    pass

if __name__ == '__main__':
    main()

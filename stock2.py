import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import PatchTSTConfig, PatchTSTForPrediction
import torch.optim as optim

class Mymodel(nn.Module):
    def __init__(self, model):
        super(Mymodel, self).__init__()
        self.patch_tst = model
        self.fc=nn.Linear(model.config.num_input_channels, 1)
        # 가정: PatchTSTForPrediction의 출력 차원이 config.output_dim으로 주어짐

    def forward(self, x):
        # PatchTST 모델을 통과시킨 후의 출력
        x = self.patch_tst(past_values=x).prediction_outputs
        # 완전연결 계층을 통과시켜 최종 출력을 얻음
        x = self.fc(x)
        return x

class Stock:
    def __init__(self, df):
        super(Stock, self).__init__()
        self.df=df.copy()
        self.scaler_all = MinMaxScaler(feature_range=(0,1))
        self.scaler_target = MinMaxScaler(feature_range=(0,1))
        self.data=df.values
        self.predictions = []  # 예측값을 저장할 리스트
        self.actuals = []  # 실제값을 저장할 리스트
        self.train_losses = []
        self.val_losses = []

    def preprocessing(self):  
        self.df['Volume'].replace(0, np.nan, inplace=True)

        # 시간대별 평균 거래량 계산
        self.df['Hour'] = pd.to_datetime(self.df.index).hour
        hourly_mean_volume = self.df.groupby('Hour')['Volume'].mean()

        # 9시의 경우, 9시의 평균 거래량으로 대체
        self.df['Volume'] = self.df.apply(lambda row: hourly_mean_volume[row['Hour']] if np.isnan(row['Volume']) else row['Volume'], axis=1)

        self.df.drop('Hour', axis=1, inplace=True)

        self.df['Volume']=self.df['Volume'].astype('int')
        return self.df

    def add_change(self, columns):
        for col in columns:
            self.df[f'{col}_chg']=self.df[col].pct_change()
        self.df.dropna(inplace=True)
        return self.df

    
    def add_col(self):
        self.df['O-C'] = self.df['Open'] - self.df['Close']
        self.df['H-L'] = self.df['High'] - self.df['Low']
        self.df['2Hr_MA'] = self.df['Close'].rolling(window=2).mean()
        self.df['4Hr_MA'] = self.df['Close'].rolling(window=4).mean()
        self.df['6Hr_MA'] = self.df['Close'].rolling(window=6).mean()
        self.df['12Hr_MA'] = self.df['Close'].rolling(window=12).mean()
        self.df['18Hr_MA'] = self.df['Close'].rolling(window=18).mean() 
        self.df['2Hr_Std'] = self.df['Close'].rolling(window = 4).std()
        self.df.dropna(inplace=True)
        return self.df

    def scale_col(self, selected_feature):
        self.selected_feature=selected_feature
        data=self.df[selected_feature].values
        self.data = self.scaler_all.fit_transform(data)
        self.scaler_target.fit_transform(data[:,0].reshape(1,-1))
        self.scaler_target.min_, self.scaler_target.scale_ = self.scaler_all.min_[0], self.scaler_all.scale_[0]


    def create_sequences(self, data, seq_length):
        xs, ys = [], []
        for i in range(len(data)-seq_length):
            x = data[i:(i+seq_length), ]
            y = data[i+seq_length, 0]  # 예측하려는 값을 0에 배치
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)
    
    def data_loader(self, seq_len, type='train'):
        self.seq_len=seq_len
        train_size = int(len(self.data) * 0.7)
        val_size = int(len(self.data) * 0.2)
        test_size = len(self.data) - train_size - val_size
        self.train_losses=[]
        self.val_losses=[]

        if type=='train':
            X, y = self.create_sequences(self.data[:train_size], seq_len)
        elif type=='valid':
            X, y = self.create_sequences(self.data[train_size:train_size+val_size], seq_len)
        elif type=='test':
            X, y = self.create_sequences(self.data[train_size+val_size:], seq_len)
        else:
            X, y = self.create_sequences(self.data, seq_len)
                
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        data = TensorDataset(X, y)
        data_loader = DataLoader(dataset=data, batch_size=16, shuffle=False)

        return data_loader
    
    def create_model(self, pred_length=1, d=0.3):
        configuration = PatchTSTConfig(prediction_length=pred_length, context_length=self.seq_len, num_input_channels=len(self.selected_feature), drop_out=d)
        self.patch = PatchTSTForPrediction(configuration)
        self.fc = nn.Linear(self.patch.config.num_input_channels, 1)
        self.model=Mymodel(self.patch) # 모델 정의
    
    def train(self, train_loader, val_loader, test_loader, patience, lr, epoch, type, min_delta=0.00001):

        # 손실 함수와 최적화 알고리즘 설정
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # 학습 과정
        epochs = epoch
        k=0.5
        if type=='train':
            patience_counter = 0
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
            
            # 학습률 감소와 Early Stopping 설정
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=patience//2, min_lr=1e-6)
            early_stopping_patience = patience
            best_val_loss = np.inf

            for epoch in range(epochs):
                self.model.train()
                train_loss = 0.0

                for seqs, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = self.model(seqs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * seqs.size(0)

                train_loss /= len(train_loader.dataset)
                self.train_losses.append(train_loss)

                self.model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for seqs, labels in val_loader:
                        outputs = self.model(seqs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item() * seqs.size(0)

                val_loss /= len(val_loader.dataset)
                self.val_losses.append(val_loss)

                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}')

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    print(f'Early stopping counter: {patience_counter}')

                # 스케줄러의 step 함수 호출
                old_lr = optimizer.param_groups[0]['lr']
                scheduler.step(val_loss)
                new_lr = optimizer.param_groups[0]['lr']

                if new_lr != old_lr:
                    print(f'Scheduler: reducing learning rate from {old_lr} to {new_lr}')


                if patience_counter >= early_stopping_patience:
                    print("Early stopping triggered")
                    break

            print(f'Best Validation Loss: {best_val_loss}')

        if type=='test':
            self.predictions=[]
            self.actuals=[]
            self.model.eval()  # 모델을 평가 모드로 설정
            test_losses = []  # 테스트 손실을 저장할 리스트

            with torch.no_grad():  # 기울기 계산을 비활성화
                for seqs, labels in test_loader:

                    outputs = self.model(seqs)

                    # 손실 계산
                    loss = criterion(outputs, labels)
                    test_losses.append(loss.item())

                    # 예측값과 실제값 저장
                    self.predictions.extend(outputs.view(-1).detach().numpy())

            return self.predictions

    def pred_value(self, type):
        if (type=='chg')|(type=='t'):
            train_size = int(len(self.data) * 0.7)
            val_size = int(len(self.data) * 0.2)
            if type=='t':
                yest=self.df.iloc[self.seq_len:,3].values.reshape(-1,1)
            else:
                yest=self.df.iloc[train_size+val_size+self.seq_len:,3].values.reshape(-1,1)
            self.predictions_inverse = np.round(self.scaler_target.inverse_transform(np.array(self.predictions).reshape(-1,1))*yest+yest, -2)
        else:
            self.predictions_inverse = np.round(self.scaler_target.inverse_transform(np.array(self.predictions).reshape(-1,1)), -2)
        return self.predictions_inverse
    
    def diff(self):
        differences = [abs(pred - actual) for pred, actual in zip(self.predictions_inverse, self.actuals_inverse)]
        print("최대 : " , max(differences) ,"최소 : " , min(differences) ,"평균: " , sum(differences) / len(differences))
        return sum(differences) / len(differences)

    def loss(self):
        # 훈련 손실과 검증 손실을 에포크별로 그래프로 그리기
        plt.figure(figsize=(10, 6))  # 그래프 크기 설정
        plt.plot(self.train_losses, label='Training Loss', marker='o')  # 훈련 손실 그래프
        plt.plot(self.val_losses, label='Validation Loss', marker='x')  # 검증 손실 그래프
        plt.title('Training and Validation Loss')  # 그래프 제목
        plt.xlabel('Epochs')  # x축 라벨
        plt.ylabel('Loss')  # y축 라벨
        plt.legend()  # 범례 표시
        plt.grid(True)  # 그리드 표시
        plt.show()  # 그래프 보여주기
    
    
    def show(self, type):
        if type=='chg':
            predictions_inverse=self.predictions_inverse[1:]
            actuals_inverse=self.actuals_inverse[1:]

        else:
            predictions_inverse=self.predictions_inverse
            actuals_inverse=self.actuals_inverse

        n = 60
        num_plots = len(predictions_inverse) // n
        if len(predictions_inverse) % n != 0:
            num_plots += 1

        # Plot 생성
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 6*num_plots))

        for i in range(num_plots):
            start_index = i * n
            end_index = min((i + 1) * n, len(predictions_inverse))
            
            # Subplot 생성
            ax = axes[i] if num_plots > 1 else axes
            
            # 예측값과 실제값 그리기
            ax.plot(predictions_inverse[start_index:end_index], label='Predictions')
            ax.plot(actuals_inverse[start_index:end_index], label='Actuals')
            ax.set_title(f'Predictions vs Actuals (Subset {i+1})')
            ax.set_xlabel('Index')
            ax.set_ylabel('Values')
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.show()
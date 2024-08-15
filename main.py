import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/Knowledge_Base/', help='Directory for data dir')
    parser.add_argument('--seed', type=int, default=42, help='Seed to split data') #42
    parser.add_argument('--model', type=str, help='Model name or path')
    # # parser.add_argument('--seeds', type=int, nargs='+', default=[42, 50, 100], help='List of seeds to split data')
    # parser.add_argument('--models', type=str, nargs='+', help='List of models to train')
    parser.add_argument('--num-classes', type=int, default=4, help='Num of grade')
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.00009, help='Learning rate') #0.0001
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Num of workers to load data')
    parser.add_argument('--phase', type=str, default='train', help='Phase: train or test')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--resume', default=False, action='store_true', help='Resume')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU')
    parser.add_argument('--path', type=str, default= f"./result") #Fix to your path to save model
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3], help='List of gpus to use')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--eval', type=str, default='test', help='Evaluation on test or eval set')
    parser.add_argument('--top-k', type=int, default=3, help='Top k accuracy')
    parser.add_argument('--experiment', type=str, default='1000_exp', help='Experiment name')
    parser.add_argument('--samples', type=int, default=100, help='Number of testing samples')
    
    return parser.parse_args()
    



# from utils import Math_Classification
# from utils import train
# from utils import evalation



if __name__== "__main__":
    args = parse_args()
    
    GPU_list = ','.join(map(str, args.gpus))
    
    import os
    os.environ['CUDA_DEVICE_ORDER'] =  'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']=  GPU_list
    print(f"Using GPU: {GPU_list}")

    from libs import *
    args.best_metric = 0
    
    if args.use_gpu and torch.cuda.is_available(): 
        device = torch.device(f'cuda')  # Change to your suitable GPU device
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    #Login
    if args.model in ['meta-llama/Llama-2-7b-hf', 'meta-llama/Meta-Llama-3-8B-Instruct']:
        from huggingface_hub import login
        login()
    
    
    if torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float16
        
        
    experiment = args.experiment
    model_name = args.model
    seed = args.seed
        

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=31) # Remember to change number of labels
    model.resize_token_embeddings(len(tokenizer))
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)   
    
    if tokenizer.pad_token is None:
        print("Adding padding token to tokenizer...")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
        tokenizer.pad_token = tokenizer.eos_token
    
    if model_name in ['Qwen/Qwen2-1.5B', 'Qwen/Qwen2-Math-1.5B', 'HuggingFaceTB/SmolLM-1.7B']:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    print(f"Training the full 310 data with model {model_name}...")
    
    
    # Load data
    df_train =pd.read_csv(f'ASDiv_data/Training_data/{seed}_train_set.csv')
    df_valid =pd.read_csv(f'ASDiv_data/Training_data/{seed}_valid_set.csv')
    df_test =pd.read_csv(f'ASDiv_data/Training_data/ASDiv-100-4th_test.csv')
    
    
    dataset_train = Dataset.from_pandas(df_train[['Question', 'label']])
    dataset_valid = Dataset.from_pandas(df_valid[['Question', 'label']])
    
    dataset_test = Dataset.from_pandas(df_test[['Question']])
    # dataset_eval = Dataset.from_pandas(df_eval)
    
    # Tokenization
    max_length = 512  # Set your fixed max length
    tokenized_dataset_train = dataset_train.map(lambda x: preprocess_function(x, tokenizer, max_length=max_length), batched=True)
    tokenized_dataset_valid = dataset_valid.map(lambda x: preprocess_function(x, tokenizer, max_length=max_length), batched=True)
    
    tokenized_dataset_test = dataset_test.map(lambda x: preprocess_function(x, tokenizer, max_length=max_length), batched=True)
    
    # Print token lengths
    def print_token_lengths(dataset, name):
        lengths = [len(x['input_ids']) for x in dataset]
        print(f"Token lengths for {name}: Min: {min(lengths)}, Max: {max(lengths)}, Avg: {np.mean(lengths):.2f}")

    print_token_lengths(tokenized_dataset_train, "Train dataset")
    print_token_lengths(tokenized_dataset_valid, "Eval dataset")
    
    print_token_lengths(tokenized_dataset_test, "Test dataset")

    # Training setup
    training_args = TrainingArguments(
    output_dir = args.path,
    learning_rate = args.lr,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    weight_decay=0.01,
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    optim="paged_adamw_32bit",
    lr_scheduler_type="linear",
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    log_level='error'
    )
    
    model.to(device)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset_train,
        eval_dataset=tokenized_dataset_valid,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    if args.phase == 'train':
        trainer.train()
        
        # Save the trained model with timestamp prefix
        model_output_dir = os.path.join(args.path, experiment, f"seed_{seed}" ,model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        trainer.save_model(model_output_dir)
        
        print(f"Model saved to {model_output_dir}")
        
        df_log = pd.DataFrame(trainer.state.log_history)

        print(df_log)
        plt.figure(figsize=(12, 6))



        # Plot validation loss
        plt.plot(df_log[['eval_loss']].dropna().reset_index(drop=True), label="Validation", color='blue')

        # Plot training loss
        plt.plot(df_log[['loss']].dropna().reset_index(drop=True), label="Train", color='red')

        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Training and Validation Losses of {args.model} through {args.epochs} epochs with seed {seed}")
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.tight_layout()

        # Save the plot as an image
        plot_output_dir = os.path.join('Loss_plot', experiment, f"seed_{seed}" , model_name)
        os.makedirs(plot_output_dir, exist_ok=True)
        
        plot_save_path = os.path.join(plot_output_dir, 'loss_plot.png')
        csv_save_path = os.path.join(plot_output_dir, 'loss_log.csv')
        
        plt.savefig(plot_save_path)
        df_log.to_csv(csv_save_path, index=False)
        print(f"Plot saved to {plot_save_path}")
        print(f"CSV saved to {csv_save_path}")

    elif args.phase == 'test':
        
        pass
    
    id_to_kc = {0: '1.OA.A.1', 1: '1.OA.A.2', 2: '2.NBT.B.5', 3: '2.NBT.B.6', 4: '2.NBT.B.7', 5: '2.OA.A.1', 6: '3.MD.D.8', 
                7: '3.NBT.A.2', 8: '3.NBT.A.3', 9: '3.NF.A.3', 10: '3.OA.A.3', 11: '3.OA.D.8', 12: '3.OA.D.9', 13: '4.MD.A.2', 
                14: '4.MD.A.3', 15: '4.NBT.B.4', 16: '4.NBT.B.5', 17: '4.NBT.B.6', 18: '4.OA.A.3', 19: '5.NBT.B.5', 20: '5.NBT.B.6', 
                21: '6.EE.B.6', 22: '6.EE.C.9', 23: '6.NS.B.4', 24: '6.RP.A.1', 25: '6.RP.A.3', 26: '6.SP.B.5', 27: '8.EE.C.8', 28: 'K.OA.A.2'}
    
    # Predictions Evaluation on Test set
    model.eval()    
    
    def predictions_output(df, tokenized_dataset):
        preds = trainer.predict(tokenized_dataset).predictions
        df_predictions = df.copy()
        if isinstance(preds, tuple):
            preds = preds[0]
        for k in range(1, args.top_k + 1):
            top_k_preds = np.argsort(preds, axis=1)[:, -k:]
            df_predictions[f'top_{k}_preds'] = list(top_k_preds)
        def map_ids_to_labels(pred_ids):
            return ','.join([ ' '+id_to_kc[i] for i in pred_ids])
        for k in range(1, args.top_k + 1):
            df_predictions[f'Top_{k}_labels'] = df_predictions[f'top_{k}_preds'].apply(map_ids_to_labels)
        
        return df_predictions
    
    df_test_predictions = predictions_output(df_test, tokenized_dataset_test)
    df_train_predictions = predictions_output(df_train, tokenized_dataset_train)
    df_valid_predictions = predictions_output(df_valid, tokenized_dataset_valid)
    
    model_name = model_name.split('/')[-1]
    
    
    # Save the predictions to CSV
    predictions_output_dir = os.path.join('Preds', experiment, f"seed_{seed}")
    os.makedirs(predictions_output_dir, exist_ok=True)
    
    
    test_predictions_csv_path = os.path.join(predictions_output_dir,f'Labels_test_{model_name}_top_k.csv')
    train_predictions_csv_path = os.path.join(predictions_output_dir, f'Labels_train_{model_name}_top_k.csv')
    valid_predictions_csv_path = os.path.join(predictions_output_dir,  f'Labels_valid_{model_name}_top_k.csv')
    
    df_test_predictions.to_csv(test_predictions_csv_path, index=False)
    print(f"Top-k predictions saved to {test_predictions_csv_path}")
    
    df_valid_predictions.to_csv(valid_predictions_csv_path, index=False)
    print(f"Top-k predictions saved to {valid_predictions_csv_path}")
    
    df_train_predictions.to_csv(train_predictions_csv_path, index=False)
    print(f"Top-k predictions saved to {train_predictions_csv_path}")    
    
    
    
    
    print(f"Evaluation on train set for seed {seed}...")   
    train_results = trainer.evaluate(eval_dataset=tokenized_dataset_train)
    print(train_results)
    
    print(f"Evaluation on Validation set for seed {seed}...")
    valid_results = trainer.evaluate(eval_dataset= tokenized_dataset_valid)
    print(valid_results)
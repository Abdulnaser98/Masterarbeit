second_approach_predictions_folder_path = "/Users/abdulnaser/Desktop/Masterarbeit/metadatatransferlearning-main/meta_tl/data/second_approach_results/predictions/"

file = []
second_approach_accuracy = []
supervised_accuracy = []
diffs = []


predicted_files = [file for file in os.listdir(second_approach_predictions_folder_path) if file.endswith('.csv')]

for predicted_file in predicted_files:
    file.append(predicted_file)
    predicted_file = pd.read_csv(os.path.join(second_approach_predictions_folder_path,predicted_file))
    predicted_file_df_processed = prepare_dataframe_supervised_approach(predicted_file)
    X = predicted_file_df_processed.iloc[:, :-2]
    y = predicted_file_df_processed.iloc[: , -2]

    # Initialize XGBoost classifier
    xgb_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Lists to store predictions and indices for each fold
    all_predictions = []
    all_indices = []
    temp_second_approach_accuracy = []
    temp_supervised_accuracy = []

    # Perform k-fold cross-validation
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Fit the model
        xgb_model.fit(X_train, y_train)

        # Make predictions
        predictions = xgb_model.predict(X_test)

        # Store predictions and indices for this fold
        all_predictions.append(predictions)
        all_indices.append(test_index)

    for fold_indices in all_indices:
        predicted_file_df_fold = predicted_file_df_processed.iloc[fold_indices][['is_match','pred']]
        temp_second_approach_accuracy.append(f1_score(predicted_file_df_fold['is_match'], predicted_file_df_fold['pred']))

    second_approach_accuracy.append(np.mean(temp_second_approach_accuracy))


    # Calculate F1-score for each fold
    f1_scores = [f1_score(y.iloc[test_index], pred) for test_index, pred in zip(all_indices, all_predictions)]
    for fold, f1 in enumerate(f1_scores, start=1):
        temp_supervised_accuracy.append(f1)

    supervised_accuracy.append(np.mean(temp_supervised_accuracy))

    # Calculate average difference
    avg_diff = average_difference(temp_second_approach_accuracy, temp_supervised_accuracy)
    diffs.append(avg_diff)


# Create DataFrame
second_appraoch_vs_superives_df = pd.DataFrame({
    'file': file,
    'second_approach_accuracy': second_approach_accuracy,
    'supervised_accuracy': supervised_accuracy,
    'avg_diff': diffs
})


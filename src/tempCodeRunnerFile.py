    # Train
    train_df = extract_features_for_split(X_train, y_train, MASK_DIR, "train")
    train_df.to_csv(DATA_DIR/"features_train.csv", index=False)
    print(f"Saved features_train.csv  ({len(train_df)} rows)\n")

    # Validation
    val_df = extract_features_for_split(X_val, y_val, MASK_DIR, "validation")
    val_df.to_csv(DATA_DIR/"features_validation.csv", index=False)
    print(f"Saved features_validation.csv  ({len(val_df)} rows)\n")

    # Test
    test_df = extract_features_for_split(X_test, y_test, MASK_DIR, "test")
    test_df.to_csv(DATA_DIR/"features_testing.csv", index=False)
    print(f"Saved features_testing.csv  ({len(test_df)} rows)\n")
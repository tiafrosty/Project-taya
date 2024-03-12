

# plot the ROC curves and CI
def get_roc_and_ci(N, k, ax, iris):
    ######3333333 ploooooooooot
    tprs = []
    all_auc = []
    base_fpr = np.linspace(0, 1, 101)
    my_model = my_models[k]['model']

    y = iris['target'].astype('int')
    X = iris.drop(['target'], axis=1)

    for i in range(N):
        s_seed = randint(1, N)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=s_seed)
        model = my_model.fit(X_train, y_train)
        #cur_auc = np.mean(roc_scores_df[i])
        if my_models[k]['label'] == 'Non-linear SVM' or my_models[k]['label'] == 'Linear SVM':
            clf = model.fit(X_train, y_train)
            calibrator = CalibratedClassifierCV(clf)
            model = calibrator.fit(X_train, y_train)
        auc = metrics.roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        #plt.plot(fpr, tpr, 'b', alpha=0.15)
        plt.sca(ax)
        #plt.sca(ax)
        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)
        all_auc.append(auc)
        #plt.plot(cur_auc, label='%s ROC (area = %0.2f)' % (m['label'], auc))
    # Custom settings for the plot
    # confidence intervals
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)
    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std
    # ploooooooooot
    plt.plot(base_fpr, mean_tprs, 'b',label=' ROC (area = %0.2f)' % (np.mean(all_auc)))
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc = 'lower right', fontsize = 15)
    #plt.title('ROC score  and CI for tested models', fontsize = 25)
    #plt.show()
    
    
    
def plot_for_every_model(my_models, data_for_plotting, p_title):

    fig, axes = plt.subplots(nrows = 3, ncols= 3)
        # for confusion matrix
    k = 0
    # PLot the classification report as a heat map for all tested models
    for i in range(len(axes)):
        for j in range(len(axes[0])):
            if data_for_plotting == 'report':
                sns.heatmap(pd.DataFrame(all_reports_sum[k]).iloc[:-1, :].T, annot=True, ax=axes[i][j], cbar=False)
            elif data_for_plotting == 'conf matrix':
                all_conf_matrix[k].plot(ax= axes[i][j])
            axes[i][j].title.set_text(my_models[k]['label'])
            # next
            k = k + 1

    fig.subplots_adjust(wspace=0.3, hspace= 0.5)
    fig.suptitle(p_title, fontsize = 25)
    plt.show()

''' UP and ARME models '''

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn')
import statsmodels.api as sm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
traindata = pd.read_table(r'/Users/ali/Desktop/Fathom Project/code/lrdata_test.txt')

# Clean data so all sales for postcode sector/district can be easily found
lrdata2 = traindata.copy()
lrdata2['Price'] = lrdata2['Price'].astype('int32')
lrdata2['Year'] = lrdata2['Year'].astype('int16')
lrdata2['Month'] = lrdata2['Month'].astype('int8')
lrdata2 = lrdata2.dropna(subset = ['Postcode'])
lrdata2 = lrdata2.dropna(subset = ['PAON'])
lrdata2['PAON'] = lrdata2['PAON'].astype(str)
lrdata2['SAON'] = lrdata2['SAON'].astype(str)
lrdata2['Postcode'] = lrdata2['Postcode'].astype(str)
pd.to_datetime(lrdata2['Date'], yearfirst=True)
uniq_pc_district = lrdata2["PC District"].unique()
lrdata2.set_index(['PC District', "PAON", "SAON", "Street"], inplace=True)
lrdata2 = lrdata2.sort_index()

'''
UP
All it is is: - ln⁡(y_(i,t) )  = M *β_t  + A *τ_i  +ϵ_(i,t)
	M: Time fixed effect dummies
	A: Individual house fixed effect dummies
	y: Sale price
'''

years = [str(year) for year in range(1995,2024)]
years = np.array(years).flatten()

# Define minimum transactions per postcode district. If under skip regression
min_transactions = 250
## to store avg gbp per post code per year
ss = pd.DataFrame(data= None, columns=uniq_pc_district, index=years)
## to store sales per post code per year
ss_n = pd.DataFrame(data= None, columns=uniq_pc_district, index=years)

rank_fail = []
ols_fail = []
progress = 0
""" Main Loop"""

# Count is here to see how many postcodes districts fail regression requirements
count_tran = 0
count_rank = 0
count_ols = 0

for dist in uniq_pc_district:
    
    progress +=1
    
    print(f"Running AVM for {dist}, ({progress}/{len(uniq_pc_district)})")
    
    area = lrdata2.loc[dist]
    area_reset = area.reset_index()

    # Drop single sales
    subset_cols = ["Type", "PAON", "SAON", "Street", "Town/City", "District", "County", "Postcode"]
    area_reset["Repeat"] = area_reset.duplicated(subset=subset_cols, keep=False)
    area1 = area_reset.copy()
    area1 = area1[area1.Repeat]

    # Skip district if sample size insufficient 
    if len(area1) < min_transactions:
        count_tran += 1
        print(f" {dist} has Insufficient sample size")
        continue
    
    area1['idx'] = range(len(area1))
    
    # Vector of each unique property in area1
    uniq_prop = area1.drop_duplicates(subset = ["Type","PAON","SAON","Street","Town/City","District","County", "Postcode"])  
    uniq_prop_idx = np.array(uniq_prop['idx'], dtype = np.int32)
    uniq_prop_label = ["v"+str(prop) for prop in uniq_prop_idx]
    
    # Log price vector 
    ln_p = np.array(np.log(area1['Price']))
    

# Matrix to store house specific effects. 
    A = np.zeros(shape= (len(area1), len(uniq_prop['idx'])), dtype=np.int32)
    
    # Loop to place '1' in rows where sale related to that property, and 0 for all other properties in the row
    counter = 0
    for i in range(len(area1)):
        if counter < len(uniq_prop_idx):   
            if i == uniq_prop_idx[counter]:
                counter +=1      
        A[i][counter-1] = 1
      
    #Matrix for time effects         
    M = pd.DataFrame(0, index = range(len(area1)), columns= years)

    for row in range(len(area1)):
        date = area1.iloc[row]['Year']
        M.iloc[row][str(date)] = 1

    del area, area1
    
    # X matrix combining A + M
    x = pd.DataFrame(A, columns= uniq_prop_label).join(M)
    # Remove any columns where all the values are 0 
    x = x.loc[:, (x != 0).any(axis=0)]
    # Drop one Year-Month column to avoid linear dependency 
    x = x.drop(columns=x.columns[len(uniq_prop_idx)])

    # Check to ensure X is full rank. If not, remove linearlly dependent columns 
    if np.linalg.matrix_rank(x)!=  min(len(x), len(x.columns)):
        print("not full rank")
        U, s, Vt = np.linalg.svd(x, full_matrices=False)
        rank = (s > 0.0001).sum()
        independent_col_indices = np.abs(Vt[rank-1]) > 0.0001
        x = x.iloc[:, independent_col_indices]
        boo = independent_col_indices[:len(uniq_prop_idx)]
        uniq_prop_idx = np.delete(uniq_prop_idx, np.where(boo == False))
     
    # Convert to X to memory-efficient containter     
    x_sparse = csr_matrix(x, dtype=float) 
    x.columns = x.columns.astype(str)
    

    # Try and run the regression
    model = LinearRegression(fit_intercept=False)
    try:
        model.fit(x_sparse, ln_p)

        results = model.coef_
        print(f'OLS running for {dist} of length {len(x)}')

        # Vector of coefficients for properties and time effects
        B_i = results[:len(uniq_prop_idx)]
        B_t = results[len(uniq_prop_idx):]

        # Average house price estimate in absence of time effects for postcode district
        avgp_i = np.mean(np.exp(B_i))
        # Average house price for district over time 
        avgp_t = np.exp(B_t)*avgp_i

        # Save results in dataframes 
        ss[dist] = ss.index.map(dict(zip(x.columns[len(uniq_prop_idx):], avgp_t)))
        ss_n[dist] = M.sum(axis=0)
    except Exception as e:
        if str(e) == "Factor is exactly singular":
            count_rank += 1
            rank_fail.append(dist)
            print(f'{dist} of length {len(x)} {str(e)}')
            ss[dist] = None
            ss_n[dist] = None
        else: 
            count_ols += 1
            ols_fail.append(dist)
            print(f'OLS failed for {dist} of length {len(x)} due to {str(e)}')
            ss[dist] = None
            ss_n[dist] = None

# interpolate missing values 
ss = ss.infer_objects(copy=False)
ss.interpolate(method='linear', axis=0, inplace=True)


print(f"{count_tran} distrcits failed minimum transactions requirement")
print(f"{count_ols} distrcits failed to run regression")
print(f"{count_rank} distrcits failed rank requirements")

#ss.to_csv(r'C:\Users\ali\Documents\AVM_folder\ss_dist12Sep.csv', index=True)
#ss_se.to_csv(r'C:\Users\ali\Documents\AVM_folder\ss_se_dist8Sep.csv', index=True)
#ss_n.to_csv(r'C:\Users\ali\Documents\AVM_folder\ss_n_dist12Sep.csv', index=True)

'''ARME model'''

## dataframes to store estimated paramters 
ss = pd.DataFrame(data= None, columns=uniq_pc_district, index=years)
ss_n = pd.DataFrame(data= None, columns=uniq_pc_district, index=years)
ss_phi_mu_msr = pd.DataFrame(data= None, columns=uniq_pc_district, index=['phi', 'mu', 'mrs'])
re_dict = {}

min_transactions = 250

ar1_fail = {}
mt_fail = []
full_rank_fail = []
not_converged = []

progress = 0


for dist in uniq_pc_district:
    
    progress +=1
    
    print(f"Running AVM for {dist}, ({progress}/{len(uniq_pc_district)})")

    ''' Step 0: Preprocessing step'''

    # Get all sales for current district 
    area = lrdata2.loc[dist]
    area1 = area.copy()
    area1 = area1.reset_index()

    # Create unique property IDs to act as the i index
    area1['property_id'] = area1['PAON'].astype(str) + '_' + area1['SAON'].astype(str) + '_' + area1['Postcode'].astype(str) + '!' + area1['Street'].astype(str) + '_' + area1['Type'].astype(str)
    area1 = area1.sort_values(by=['property_id', 'Date'])

    # Remove repeat sales of the same property within the same year as this messes up things
    area1 = area1.drop_duplicates(subset=['property_id', 'Year'], keep='first')

    # Number which sale of property i the row belongs to. Equivelent to j in the paper 
    area1['sale_number'] = area1.groupby('property_id').cumcount() + 1
    area1.reset_index(drop=True, inplace=True)
    
    # Skip this district/sector if the sample size is too small 
    if len(area1) < min_transactions:
        mt_fail.append(dist)
        print(f"{dist} failed minimum transactions requirement")   
        ss[dist] = None
        ss_n[dist] = None
        ss_phi_mu_msr[dist] = None
        continue 

    # Create shifted versions of 'Year' and 'property_id' to see if the sale below belongs to same property as current in current row.
    # Used to calculate gamma
    area1['Next_Year'] = area1['Year'].shift(-1)
    area1['Next_property_id'] = area1['property_id'].shift(-1)

    # Calculate gamma only for same properties
    area1['gamma'] = np.where(area1['property_id'] == area1['Next_property_id'], 
                            area1['Next_Year'] - area1['Year'], 0)

    # Create X and y
    X = pd.get_dummies(area1['Year'], dtype=float)
    X = X.drop(columns=X.columns[0])
    X.insert(0,'mu', 1.0)
    y = np.array(np.log(area1['Price']))

    N = len(y)
    # r is an N x N diagonal matrix that stores error term variances
    r = np.zeros((N,N))


    # Index of first sales, j = 1
    mask_first_sale = area1['sale_number'] == 1

    # diagonal terms where j = 1 are given 1s
    r[np.arange(N)[mask_first_sale], np.arange(N)[mask_first_sale]] = 1
    mask_same_property = area1['property_id'] == area1['Next_property_id']

    # Index for j > 1
    mask_subsequent_sales = ~mask_first_sale

    # The associated gamma is given in the previous sale of the property where j>1
    gamma_pos = np.arange(N)[mask_subsequent_sales] - 1
    
    # Random effects are the individual properties 
    groups = area1['property_id'].to_numpy()
    Z = pd.get_dummies(groups, dtype = float)
    unique_properties = Z.columns


    try:
        
        ''' Step 1. regress y on X to with group random effects to obtain estimate of residuals '''
        
        model = sm.MixedLM(y, X, groups).fit(reml=False) 

        resid0 = model.resid

        sigma_sqrd_eps = model.scale

        ''' Step 2. estimate phi from obtained residuals using non-linear least squares '''


        def estimate_phi(residuals):
            
            # Extract residuals for sales j where j > 1  *** Only repeat sales have correlated errors ***
            resid_j = residuals[mask_subsequent_sales]

            # Extract the associated residuals for sales j-1
            resid_j_1 = residuals[gamma_pos]

            # Extract gamma for sales j where j > 1
            gamma_vec = area1['gamma'][gamma_pos].values

            #function to estimate phi from: u_j = (phi^gamma) * u_(j-1) + eps
            def nls_fun(resid_j_i, phi, gamma):
                fun_est =  resid_j_1 * np.power(phi,gamma)
                return fun_est
            
            # estimate phi using nls
            popt, pcov = curve_fit(lambda resid_j_1, phi: nls_fun(resid_j_1, phi, gamma_vec), resid_j_1, resid_j, p0=[0.99], bounds = (0,1))
            
            phi_est = popt[0]
            return phi_est

        
        ''' Step 3: regress Ty on TX to obtian mu and beta estimates. Then repeat regression in step 1 using these estimates to obtain residuals '''

        
        def estimate_resid(phi, sigma_sqrd_eps):

                    # create and populate T 
                    T = np.identity(N, dtype=np.float64)
                    T[mask_subsequent_sales, gamma_pos ] = -np.power(phi, area1['gamma'][gamma_pos])
                    
                    # Diagonal element of r is 1-phi^(phi*gamma) if j > 1, and 1 otherwise. 
                    r[np.arange(N)[mask_subsequent_sales], np.arange(N)[mask_subsequent_sales]] = 1 - np.power(phi, 2 * area1['gamma'][gamma_pos])
                    
                    # Estimated variance-covariance matrix of the erros 
                    V = (sigma_sqrd_eps/(1-phi**2)) * r 
                    
                    # P = estimated weights to premultiply to Ty and TX to correct for the heteroscedasticity 
                    P = np.diag(1 / np.sqrt(np.diag(V)))

                    Ty = T @ y 
                    TX = T @ X

                    PTy = P @ Ty
                    PTX = P @ TX

                    # regress Ty on TX 
                    T_model = sm.MixedLM(PTy, PTX, groups)
                    T_results = T_model.fit(maxiter = 1000)

                    # sigma^2_eps estimate 
                    sigma2_eps = T_results.scale
                    
                    beta = T_results.params[:-1]
                    re_effects = T_results.random_effects   
                    tau = np.array([re_effects[prop] for prop in unique_properties]).ravel()
                    Ztau = Z @ tau

                    # calculate the residuals of the untransformed model using estimated beta, mu and tau 
                    residuals = y - X @ beta
                    residuals = residuals - Ztau
                    
                    return residuals, sigma2_eps


        ''' Step 4: repeat steps 1 - 3 until phi converges '''
        while True:
            p_est = estimate_phi(resid0)
            resid0, sigma_sqrd_eps = estimate_resid(p_est, sigma_sqrd_eps)
            p_est2 = estimate_phi(resid0)
            break 

        
        ''' Step 5: Estimate final paramter values with obtianed value of phi and simga-squared. 
            With phi and sigma-squared, the variance-covariance matrix of eps can be defined. Therefore.
            the heteroscedasticity in eps can be corrected. The transformation of y* = Ty and 
            X* = TX allows for the removal autocorrelated errors, applying a similar transformation
            to the Prais-Winston transformation.  '''
        
        phi = p_est2 

        # Fill in variance-covaraince matrix diagonals 
        r[np.arange(N)[mask_subsequent_sales], np.arange(N)[mask_subsequent_sales]] = 1 - np.power(phi, 2 * area1['gamma'][gamma_pos])

        # Transformation matrix to remove serial correlation 
        T = np.identity(N, dtype=np.float64)
        T[mask_subsequent_sales, gamma_pos ] = -np.power(phi, area1['gamma'][gamma_pos])

        # Var-cov matrix
        V = (sigma_sqrd_eps/(1-phi**2)) * r 

        P = np.diag(1 / np.sqrt(np.diag(V)))

        Ty = T @ y 
        TX = T @ X
        PTy = P @ Ty
        PTX = P @ TX

        # regress PTy on PTX 
        fin_model = sm.MixedLM(PTy, PTX, groups)
        fin_results = fin_model.fit(reml=False, full_output = True, maxiter=1000)

        sigma2_eps = fin_results.scale

        betas = fin_results.params[:-1]

        re_effects = fin_results.random_effects  

        fitted_vals = fin_results.fittedvalues
        residuals = fin_results.resid 

        msr = sigma2_eps + fin_results.cov_re
        mu = betas['mu']
        
        ss_phi_mu_msr[dist] = [phi,mu,msr]
        ss[dist] = betas[1:]

        # Extract random effects and save them in dictionary 
        re = {}
        for key, value in fin_results.random_effects.items():
            new_key = "".join(key.split("!")[0])  
            group_var = value.get('Group Var', None)  
            re[new_key] = group_var
        re_dict[dist] = re

        converged = fin_results.converged

        print('***************************************************************')
        if converged == True:
            print(f"Model convergence successful for {dist} of length {len(Ty)}")
        else:
            not_converged.append(dist)
            print(f"Model failed to converge for {dist} of length {len(Ty)}")
        print('***************************************************************')

    except Exception as e:
        ar1_fail[dist] = str(e)
        print(f'Estimation failed for {dist} of length {len(Ty)} due to {str(e)}')
        ss[dist] = None
        ss_n[dist] = None
        ss_phi_mu_msr[dist] = None
    
        
ss = ss.infer_objects(copy=False)
ss.interpolate(method='linear', axis=0, inplace=True)


print(f"{len(mt_fail)} distrcits failed minimum transactions requirement")
print(f"{len(ar1_fail.keys())} distrcits failed to run regression")
print(f"{len(not_converged)} districts did not converge")


        

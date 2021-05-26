import pandas as pd
import os
import glob
#seduto senza supporto
#dati torace
path1t = r'C:\Users\chiar\Desktop\rawsignals\A\T' # use your path
all_files1t = glob.glob(os.path.join(path1t, "*.csv"))
name= ['quat_1t', 'quat_2t', 'quat_3t', 'quat_4t', 'user']
a_t = pd.concat((pd.read_csv(f, names=name) for f in all_files1t), ignore_index=True)
a_t.drop('user', inplace=True, axis=1)
#dati addome
path1a = r'C:\Users\chiar\Desktop\rawsignals\A\A' # use your path
all_files1a = glob.glob(os.path.join(path1a, "*.csv"))
name= ['quat_1a', 'quat_2a', 'quat_3a', 'quat_4a', 'user']
a_a = pd.concat((pd.read_csv(f, names=name) for f in all_files1a), ignore_index=True)
a_a.drop('user', inplace=True, axis=1)
#dati reference
path1r = r'C:\Users\chiar\Desktop\rawsignals\A\R' # use your path
all_files1r = glob.glob(os.path.join(path1r, "*.csv"))
name= ['quat_1r', 'quat_2r', 'quat_3r', 'quat_4r', 'user']
a_r = pd.concat((pd.read_csv(f, names=name) for f in all_files1r), ignore_index=True)
a_tot = pd.concat([a_t, a_a, a_r], axis=1)
#a
a_tot['activity'] = 'sitting'
#seduto con supporto
path2t = r'C:\Users\chiar\Desktop\rawsignals\B\T' # use your path
all_files2t = glob.glob(os.path.join(path2t, "*.csv"))
name= ['quat_1t', 'quat_2t', 'quat_3t', 'quat_4t', 'user']
b_t = pd.concat((pd.read_csv(f, names=name) for f in all_files2t), ignore_index=True)
b_t.drop('user', inplace=True, axis=1)

path2a = r'C:\Users\chiar\Desktop\rawsignals\B\A' # use your path
all_files2a = glob.glob(os.path.join(path2a, "*.csv"))
name= ['quat_1a', 'quat_2a', 'quat_3a', 'quat_4a', 'user']
b_a = pd.concat((pd.read_csv(f, names=name) for f in all_files2a), ignore_index=True)
b_a.drop('user', inplace=True, axis=1)
path2r = r'C:\Users\chiar\Desktop\rawsignals\B\R' # use your path
all_files2r = glob.glob(os.path.join(path2r, "*.csv"))
name= ['quat_1r', 'quat_2r', 'quat_3r', 'quat_4r', 'user']
b_r = pd.concat((pd.read_csv(f, names=name) for f in all_files2r), ignore_index=True)
b_tot = pd.concat([b_t, b_a, b_r], axis=1)
b_tot['activity'] = 'sitting'
#supino
path3t = r'C:\Users\chiar\Desktop\rawsignals\C\T' # use your path
all_files3t = glob.glob(os.path.join(path3t, "*.csv"))
name= ['quat_1t', 'quat_2t', 'quat_3t', 'quat_4t', 'user']
c_t = pd.concat((pd.read_csv(f, names=name) for f in all_files3t), ignore_index=True)
c_t.drop('user', inplace=True, axis=1)
path3a = r'C:\Users\chiar\Desktop\rawsignals\C\A' # use your path
all_files3a = glob.glob(os.path.join(path3a, "*.csv"))
name= ['quat_1a', 'quat_2a', 'quat_3a', 'quat_4a', 'user']
c_a = pd.concat((pd.read_csv(f, names=name) for f in all_files3a), ignore_index=True)
c_a.drop('user', inplace=True, axis=1)
path3r = r'C:\Users\chiar\Desktop\rawsignals\C\R' # use your path
all_files3r = glob.glob(os.path.join(path3r, "*.csv"))
name= ['quat_1r', 'quat_2r', 'quat_3r', 'quat_4r', 'user']
c_r = pd.concat((pd.read_csv(f, names=name) for f in all_files3r), ignore_index=True)
c_tot = pd.concat([c_t, c_a,c_r], axis=1)
c_tot['activity'] = 'supine'
#prono
path4t = r'C:\Users\chiar\Desktop\rawsignals\D\T' # use your path
all_files4t = glob.glob(os.path.join(path4t, "*.csv"))
name= ['quat_1t', 'quat_2t', 'quat_3t', 'quat_4t', 'user']
d_t = pd.concat((pd.read_csv(f, names=name) for f in all_files4t), ignore_index=True)
d_t.drop('user', inplace=True, axis=1)
path4a = r'C:\Users\chiar\Desktop\rawsignals\D\A' # use your path
all_files4a = glob.glob(os.path.join(path4a, "*.csv"))
name= ['quat_1a', 'quat_2a', 'quat_3a', 'quat_4a', 'user']
d_a = pd.concat((pd.read_csv(f, names=name) for f in all_files4a), ignore_index=True)
d_a.drop('user', inplace=True, axis=1)
path4r = r'C:\Users\chiar\Desktop\rawsignals\D\R' # use your path
all_files4r = glob.glob(os.path.join(path4r, "*.csv"))
name= ['quat_1r', 'quat_2r', 'quat_3r', 'quat_4r', 'user']
d_r = pd.concat((pd.read_csv(f, names=name) for f in all_files4r), ignore_index=True)
d_tot = pd.concat([d_t, d_a, d_r], axis=1)
d_tot['activity'] = 'prone'
#sx
path5t = r'C:\Users\chiar\Desktop\rawsignals\E\T' # use your path
all_files5t = glob.glob(os.path.join(path5t, "*.csv"))
name= ['quat_1t', 'quat_2t', 'quat_3t', 'quat_4t', 'user']
e_t = pd.concat((pd.read_csv(f, names=name) for f in all_files5t), ignore_index=True)
e_t.drop('user', inplace=True, axis=1)
path5a = r'C:\Users\chiar\Desktop\rawsignals\E\A' # use your path
all_files5a = glob.glob(os.path.join(path5a, "*.csv"))
name= ['quat_1a', 'quat_2a', 'quat_3a', 'quat_4a', 'user']
e_a = pd.concat((pd.read_csv(f, names=name) for f in all_files5a), ignore_index=True)
e_a.drop('user', inplace=True, axis=1)
path5r = r'C:\Users\chiar\Desktop\rawsignals\E\R' # use your path
all_files5r = glob.glob(os.path.join(path5r, "*.csv"))
name= ['quat_1r', 'quat_2r', 'quat_3r', 'quat_4r', 'user']
e_r = pd.concat((pd.read_csv(f, names=name) for f in all_files5r), ignore_index=True)
e_tot = pd.concat([e_t, e_a,e_r], axis=1)
e_tot['activity'] = 'lying_left'
#dx
path6t = r'C:\Users\chiar\Desktop\rawsignals\F\T' # use your path
all_files6t = glob.glob(os.path.join(path6t, "*.csv"))
name= ['quat_1t', 'quat_2t', 'quat_3t', 'quat_4t', 'user']
f_t = pd.concat((pd.read_csv(f, names=name) for f in all_files6t), ignore_index=True)
f_t.drop('user', inplace=True, axis=1)
path6a = r'C:\Users\chiar\Desktop\rawsignals\F\A' # use your path
all_files6a = glob.glob(os.path.join(path6a, "*.csv"))
name= ['quat_1a', 'quat_2a', 'quat_3a', 'quat_4a', 'user']
f_a = pd.concat((pd.read_csv(f, names=name) for f in all_files6a), ignore_index=True)
f_a.drop('user', inplace=True, axis=1)
path6r = r'C:\Users\chiar\Desktop\rawsignals\F\R' # use your path
all_files6r = glob.glob(os.path.join(path6r, "*.csv"))
name= ['quat_1r', 'quat_2r', 'quat_3r', 'quat_4r', 'user']
f_r = pd.concat((pd.read_csv(f, names=name) for f in all_files6r), ignore_index=True)
f_tot = pd.concat([f_t, f_a,f_r], axis=1)
f_tot['activity'] = 'lying_right'
#in piedi
path7t = r'C:\Users\chiar\Desktop\rawsignals\G\T' # use your path
all_files7t = glob.glob(os.path.join(path7t, "*.csv"))
name= ['quat_1t', 'quat_2t', 'quat_3t', 'quat_4t', 'user']
g_t = pd.concat((pd.read_csv(f, names=name) for f in all_files7t), ignore_index=True)
g_t.drop('user', inplace=True, axis=1)
path7a = r'C:\Users\chiar\Desktop\rawsignals\G\A' # use your path
all_files7a = glob.glob(os.path.join(path7a, "*.csv"))
name= ['quat_1a', 'quat_2a', 'quat_3a', 'quat_4a', 'user']
g_a = pd.concat((pd.read_csv(f, names=name) for f in all_files7a), ignore_index=True)
g_a.drop('user', inplace=True, axis=1)
path7r = r'C:\Users\chiar\Desktop\rawsignals\G\R' # use your path
all_files7r = glob.glob(os.path.join(path7r, "*.csv"))
name= ['quat_1r', 'quat_2r', 'quat_3r', 'quat_4r', 'user']
g_r = pd.concat((pd.read_csv(f, names=name) for f in all_files7r), ignore_index=True)
g_tot = pd.concat([g_t, g_a,g_r], axis=1)
g_tot['activity'] = 'standing'
#scale
path8t = r'C:\Users\chiar\Desktop\rawsignals\I\T' # use your path
all_files8t = glob.glob(os.path.join(path8t, "*.csv"))
name= ['quat_1t', 'quat_2t', 'quat_3t', 'quat_4t', 'user']
i_t = pd.concat((pd.read_csv(f, names=name) for f in all_files8t), ignore_index=True)
i_t.drop('user', inplace=True, axis=1)
path8a = r'C:\Users\chiar\Desktop\rawsignals\I\A' # use your path
all_files8a = glob.glob(os.path.join(path8a, "*.csv"))
name= ['quat_1a', 'quat_2a', 'quat_3a', 'quat_4a', 'user']
i_a = pd.concat((pd.read_csv(f, names=name) for f in all_files8a), ignore_index=True)
i_a.drop('user', inplace=True, axis=1)
path8r = r'C:\Users\chiar\Desktop\rawsignals\I\R' # use your path
all_files8r = glob.glob(os.path.join(path8r, "*.csv"))
name= ['quat_1r', 'quat_2r', 'quat_3r', 'quat_4r', 'user']
i_r = pd.concat((pd.read_csv(f, names=name) for f in all_files8r), ignore_index=True)
i_tot = pd.concat([i_t, i_a,i_r], axis=1)
i_tot['activity'] = 'stairs'
#camminata lenta
path9t = r'C:\Users\chiar\Desktop\rawsignals\L\T'  # use your path
all_files9t = glob.glob(os.path.join(path9t, "*.csv"))
name = ['quat_1t', 'quat_2t', 'quat_3t', 'quat_4t', 'user']
l_t = pd.concat((pd.read_csv(f, names=name) for f in all_files8t), ignore_index=True)
l_t.drop('user', inplace=True, axis=1)
path9a = r'C:\Users\chiar\Desktop\rawsignals\L\A' # use your path
all_files8a = glob.glob(os.path.join(path8a, "*.csv"))
name= ['quat_1a', 'quat_2a', 'quat_3a', 'quat_4a', 'user']
l_a = pd.concat((pd.read_csv(f, names=name) for f in all_files8a), ignore_index=True)
l_a.drop('user', inplace=True, axis=1)
path9r = r'C:\Users\chiar\Desktop\rawsignals\L\R' # use your path
all_files9r = glob.glob(os.path.join(path9r, "*.csv"))
name= ['quat_1r', 'quat_2r', 'quat_3r', 'quat_4r', 'user']
l_r = pd.concat((pd.read_csv(f, names=name) for f in all_files9r), ignore_index=True)
l_tot = pd.concat([l_t, l_a,l_r], axis=1)
l_tot['activity'] = 'walking'
#camminata veloce
path12t = r'C:\Users\chiar\Desktop\rawsignals\M\T'  # use your path
all_files12t = glob.glob(os.path.join(path9t, "*.csv"))
name = ['quat_1t', 'quat_2t', 'quat_3t', 'quat_4t', 'user']
m_t = pd.concat((pd.read_csv(f, names=name) for f in all_files12t), ignore_index=True)
m_t.drop('user', inplace=True, axis=1)
path12a = r'C:\Users\chiar\Desktop\rawsignals\M\A' # use your path
all_files12a = glob.glob(os.path.join(path8a, "*.csv"))
name= ['quat_1a', 'quat_2a', 'quat_3a', 'quat_4a', 'user']
m_a = pd.concat((pd.read_csv(f, names=name) for f in all_files12a), ignore_index=True)
m_a.drop('user', inplace=True, axis=1)
path12r = r'C:\Users\chiar\Desktop\rawsignals\M\R' # use your path
all_files12r = glob.glob(os.path.join(path12r, "*.csv"))
name= ['quat_1r', 'quat_2r', 'quat_3r', 'quat_4r', 'user']
m_r = pd.concat((pd.read_csv(f, names=name) for f in all_files12r), ignore_index=True)
m_tot = pd.concat([m_t, m_a,m_r], axis=1)
m_tot['activity'] = 'walking'
#corsa
path10t = r'C:\Users\chiar\Desktop\rawsignals\N\T' # use your path
all_files10t = glob.glob(os.path.join(path10t, "*.csv"))
name= ['quat_1t', 'quat_2t', 'quat_3t', 'quat_4t', 'user']
n_t = pd.concat((pd.read_csv(f, names=name) for f in all_files10t), ignore_index=True)
n_t.drop('user', inplace=True, axis=1)
path10a = r'C:\Users\chiar\Desktop\rawsignals\N\A' # use your path
all_files10a = glob.glob(os.path.join(path10a, "*.csv"))
name= ['quat_1a', 'quat_2a', 'quat_3a', 'quat_4a', 'user']
n_a = pd.concat((pd.read_csv(f, names=name) for f in all_files10a), ignore_index=True)
n_a.drop('user', inplace=True, axis=1)
path10r = r'C:\Users\chiar\Desktop\rawsignals\N\R' # use your path
all_files10r = glob.glob(os.path.join(path10r, "*.csv"))
name= ['quat_1r', 'quat_2r', 'quat_3r', 'quat_4r', 'user']
n_r = pd.concat((pd.read_csv(f, names=name) for f in all_files10r), ignore_index=True)
n_tot = pd.concat([n_t, n_a,n_r], axis=1)
n_tot['activity'] = 'running'
#cyclette
path11t = r'C:\Users\chiar\Desktop\rawsignals\O\T' # use your path
all_files11t = glob.glob(os.path.join(path11t, "*.csv"))
name= ['quat_1t', 'quat_2t', 'quat_3t', 'quat_4t', 'user']
o_t = pd.concat((pd.read_csv(f, names=name) for f in all_files11t), ignore_index=True)
o_t.drop('user', inplace=True, axis=1)
path11a = r'C:\Users\chiar\Desktop\rawsignals\O\A' # use your path
all_files11a = glob.glob(os.path.join(path11a, "*.csv"))
name= ['quat_1a', 'quat_2a', 'quat_3a', 'quat_4a', 'user']
o_a = pd.concat((pd.read_csv(f, names=name) for f in all_files11a), ignore_index=True)
o_a.drop('user', inplace=True, axis=1)
path11r = r'C:\Users\chiar\Desktop\rawsignals\O\R' # use your path
all_files11r = glob.glob(os.path.join(path11r, "*.csv"))
name= ['quat_1r', 'quat_2r', 'quat_3r', 'quat_4r', 'user']
o_r = pd.concat((pd.read_csv(f, names=name) for f in all_files11r), ignore_index=True)
o_tot = pd.concat([o_t, o_a, o_r], axis=1)
o_tot['activity'] = 'cyclette'

frames_tot = [a_tot, b_tot, c_tot, d_tot, e_tot, f_tot, g_tot, i_tot, l_tot, m_tot, n_tot, o_tot]
dataset_tot = pd.concat(frames_tot, join='outer')
dataset_tot.to_csv("tot_complete.csv", index = False)
#Calculate interference visibility of |0>+|1> states with weak coherent states
#Get the superposition from the actual OM-interaction. Include initial thermal occupation

import matplotlib.pyplot as plt
import numpy as np
from qutip import *

def mixed_sup(nth, scattering, wcs1_amp, wcs2_amp):
	#Calculate with all the heating present in the mechanical state from the beginning
	#Mach powers to the respecting powers in the optical mode for blue and mechanival mode for red

	N = 10                    # Number of mech states

	## 2-Mode operators for optics and mechanics
	o_op = tensor(destroy(N), qeye(N))
	m_op = tensor(qeye(N), destroy(N))
	num_m = m_op.dag() * m_op
	num_o = o_op.dag() * o_op

	#first simulate the scattering probability
	#set up initial state
	opt_mode = coherent_dm(N, 0.)
	mech_mode = thermal_dm(N, nth)
	rho_0 = tensor(opt_mode, mech_mode)

	H_om = Qobj.expm( 1j*np.sqrt(scattering) * (m_op.dag()*o_op.dag() + m_op*o_op) )
	rho_1 = H_om.dag() * rho_0 * H_om

	#sanity checks
	scattering_check = expect(num_o, rho_1)
	print('\n\nscattering prob ' + str(scattering_check)[0:6])

	if wcs1_amp == 0:
		wcs1_amp = np.sqrt(scattering_check)
	print('Calculating with n_wcs1 = ' + str(wcs1_amp**2)[0:6])
	wcs1_mode = coherent_dm(N, wcs1_amp)		#create 50:50 superposition

	#enlarge Hilbert space, call vectors 'optics_a, optics_b, mechanics'
	rho_2 = tensor(wcs1_mode, rho_1)
	#operators
	oa_op = tensor(destroy(N), qeye(N), qeye(N))
	ob_op = tensor(qeye(N), destroy(N), qeye(N))

	#sanity checks
	nopt11 = expect(oa_op.dag()*oa_op, rho_2)
	nopt12 = expect(ob_op.dag()*ob_op, rho_2)
	print('Blue before BS na = ' + str(nopt11) + ' nb = ' + str(nopt12)[0:6] + ' sum = ' + str(nopt11+nopt12)[0:6])

	#BS
	theta = np.pi / 2	#BS transmission
	BS_A = Qobj.expm( 1j/2*theta * (oa_op.dag()*ob_op + oa_op*ob_op.dag()) )
	rho_3 = BS_A.dag() * rho_2 * BS_A

	#sanity checks
	nopt21 = expect(oa_op.dag()*oa_op, rho_3)
	nopt22 = expect(ob_op.dag()*ob_op, rho_3)
	print('Blue after BS na = ' + str(np.real(nopt21))[0:6] + ' nb = ' + str(np.real(nopt22))[0:6] + ' sum = ' + str(np.real(nopt21+nopt22))[0:6])

	#project states onto n=1 and 2 fock basis of optics (heralding)
	proj = tensor((basis(N, 1) * basis(N, 1).dag() + basis(N, 2) * basis(N, 2).dag() + basis(N, 3) * basis(N, 3).dag()), qeye(N), qeye(N))

	rho_4 = proj.dag() * rho_3 * proj
	rho_4 = rho_4.unit()

	#trace out optics, leave only mechanics
	rho_4 = rho_4.ptrace(2).unit()

	#sanity
	n_mech = expect(num(N), rho_4)
	print('Mechanical excitation n = ' + str(n_mech)[0:6])

	#now the second interferencestep
	#maximum visibility is attained for angle pi/2
	angle = np.pi/2

	wcs2_amp = np.sqrt(n_mech) * wcs2_amp

	wcs2_mode = ket2dm(coherent(N, wcs2_amp*np.exp(-1j*angle)))

	#Beamsplitter input modes
	rho_5 = tensor(wcs2_mode, rho_4)

	#sanity checks
	nopt11 = expect(tensor(num(N), qeye(N)), rho_5)
	nopt12 = expect(tensor(qeye(N), num(N)), rho_5)
	print('Red before BS2 na = ' + str(nopt11)[0:6] + ' nb = ' + str(nopt12)[0:6] + ' sum = ' + str(nopt11+nopt12)[0:6])

	#Beamsplitter transmission
	T = 0.5

	#Calculate mode in one output arm of beamsplitter
	n_out1 = np.real(T*expect(num_m, rho_5) + (1-T)*expect(num_o, rho_5) + 1j* np.sqrt(T)*np.sqrt(1-T) * expect(o_op.dag()*m_op-o_op*m_op.dag(), rho_5))

	n_out2 = np.real((1-T)*expect(num_m, rho_5) + T*expect(num_o, rho_5) - 1j* np.sqrt(T)*np.sqrt(1-T) * expect(o_op.dag()*m_op-o_op*m_op.dag(), rho_5))

	print('Red after BS2 na = ' + str(n_out1)[0:6] + ' nb = ' + str(n_out2)[0:6] + ' sum = ' + str(n_out1+n_out2)[0:6])

	vis = (n_out1-n_out2) / (n_out1+n_out2)
	print('Max vis V = ' + str(100 * vis)[0:6] + '%')
	return [np.real(n_mech), np.real(vis)]


##
pts1 = 31
pts2 = 4
wcs2_amp = np.linspace(0.5, 2.0, pts1)
ntherm = [0.00, 0.05, 0.15, 0.30]
v = np.zeros([pts2, pts1])
n = np.zeros([pts2, pts1])

ax1 = plt.subplot(2, 1, 1)
#ax2 = plt.subplot(2, 1, 2)
#ax1.plot([0.002, 0.002], [0., 1.])
#ax2.plot([0.002, 0.002], [0., 2.])

for jj in range(pts2):
	for ii in range(pts1):
		[n[jj, ii], v[jj, ii]] = mixed_sup(ntherm[jj], 0.002, 0., wcs2_amp[ii])

	ax1.plot(wcs2_amp, v[jj, :])


np.savetxt('red_vis.dat', v)

ax1.set_xlabel('n_wcs_red / n_om_red')
ax1.set_ylabel('vis')
#ax2.set_ylabel('n_herald')
plt.legend(['n_th=0.00', 'n_th=0.05', 'n_th=0.15', 'n_th=0.30' ])
plt.show()

print('done')

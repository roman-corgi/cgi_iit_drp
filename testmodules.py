imports = ['gsw_testing', 'cal', 'proc_cgi_frame', 'emccd_detect']
imports = sorted(imports)

modules = {}
for x in imports:
	print('----', x, '----')
	try:
		modules[x] = __import__(x)
		print(x,"successfully imported")
	except ImportError:
		print("Error importing", x)
	
	try:
		print(x, ': ', modules[x].__version__)
	except:
		pass
		
	try:
		print("  ", modules[x].__path__)
	except:
		pass
		
	print(" ")

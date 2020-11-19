## Condensation-model (work in progress)
implementation of a condensation model for comparison with experimental data

### Calculations based on 
Eimann et al. DOI: 10.1016/j.ijheatmasstransfer.2020.119734


### Further sources: 
- Caruso et al. DOI: 10.1016/j.ijheatmasstransfer.2013.09.049
- Sommers et al. DOI: 10.1016/j.expthermflusci.2012.01.031
- Brouwers H.J.H. Effect of Fog Formation on Turbulent Vapor Condensation With Noncondensable Gases, J. Heat Transfer 118, p. 243-245, 1996, 
- El Sherbini, Jacobi DOI: 10.1016/j.jcis.2006.02.018
- Peterson et al. DOI: 10.1115/1.2911397

### Material data based on:
Tsilingiris et al. DOI: 10.1016/j.enconman.2007.09.015
and the CoolProp Module DOI: 10.1021/ie4033999 (http://www.coolprop.org/index.html)


### Tests
``` python3 test.py ```

### How To
heat transfer model can be calculated by config file only or config file and data file.
data file should be space or comma separated value table and has to be entered in
```eimann2020.py```
geometrical info and contact angles have to be specified in the config file.

critical droplet radius is calculated from droplet force balance and entered into heat transfer calculation.
it can also be overwritten manually to match experimental data.

results are output into a specified output file

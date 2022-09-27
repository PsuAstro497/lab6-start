### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 8be9bf52-a0a3-11ec-045f-3962ad227049
begin
	using CSV, DataFrames, Query
	using StatsBase: mean, std
	using Optim, ForwardDiff#, NLSolversBase
	using Turing, Distributions, MCMCChains
	using LinearAlgebra,PDMats
	using Plots, LaTeXStrings, Plots.Measures
	using PlutoUI, PlutoTeachingTools
	using Downloads
	#using ParameterHandling
	using Random
	Random.seed!(123)
end

# ╔═╡ 82d5eb4f-5724-4c72-b6e0-f6d5fc7f4313
md"""
**Astro 497:  Lab 6, Ex 1**
# Explanatory Data Analysis: $br Measuring Planet Masses
"""

# ╔═╡ 57141374-dd5a-4eaa-8235-b2310ef2d600
TableOfContents()

# ╔═╡ 921f13df-bc87-4d1f-8429-90cd234a65a1
md"""
In this lab, you'll analyze radial velocity (RV) observations of 51 Pegasi, the first sun-like star discovered to host an exoplanet.  First, you'll fit a Keplerian model to data from one observatory/instrument.  Then you'll use both bootstrap and MCMC methods to obtain estimates of the uncertainties in the RV amplitude parameter, $K$, which is proportional to the planet's mass.  Then you'll repeat the analysis using data from a second observatory/instrument.  You'll compare the uncertainty estimates from each method and instrument, so as to develop intuition for how measurement uncertainties from different methods and datasets compare.
"""

# ╔═╡ 4ce8dd30-9b3e-4411-9a43-dcd77149aea2
md"""
## Keplerian RV Model
First, let's remind ourselves of the Keplerian model that you'll fit to the data.

$$\Delta RV(t) = \frac{K}{\sqrt{1-e^2}} \left[\cos(\omega+ν(t)) + e \cos(\omega) \right]$$

```math
\begin{eqnarray}
K & = & \frac{2\pi a \sin i}{P} \frac{m_{pl}}{M_\star+m_{pl}} \\
%  & = & \frac{2\pi a \sin i}{P} \\
\end{eqnarray}
```

$$\tan\left(\frac{\nu(t)}{2}\right) = \sqrt{\frac{1+e}{1-e}} \tan\left(\frac{E(t)}{2}\right)$$

The true anomaly ($\nu(t)$ or $f(t)$ or $T(t)$) specifies position in orbit using angle from focus of ellipse

The eccentric anomaly ($E(t)$) specifies the position in orbit using the angle from center of elipse and is computed via Kepler's equation
$$M(t) = E(t) - e \sin(E(t))$$
"""

# ╔═╡ 6e34f8e1-99cb-4557-9537-2e33ee864267
md"""
The mean anomaly ($M(t)$) increases linearly with time
$$M(t) = \frac{2π t}{P} + M_0.$$

You can visualize the RV curve predicted by the Keplerian model below.
"""

# ╔═╡ 4bdcca25-c37f-4079-b222-be773adc2b8f
md"### Interactive Keplerian RV Model"

# ╔═╡ 2306a2d5-2924-45e0-adec-b90d536d2949
md"""
P: $(@bind P_plt NumberField(1:0.1:100, default=4))
K: $(@bind K_plt NumberField(0:0.1:100, default=30))
e: $(@bind e_plt NumberField(0:0.05:1, default=0))
ω: $(@bind ω_plt NumberField(0:0.05:2π, default=0))
M₀: $(@bind M0_plt NumberField(0:0.05:2π, default=0))
"""

# ╔═╡ 3d1821a6-f134-49d6-a4b0-39d6d28ab420
md"""
# Ingest & select RV data to be analyzed
"""

# ╔═╡ 9945831b-96ca-4eb4-8993-4acb0dc4b08e
md"""
Soon, we'll begin analyzing the observations of 51 Pegasi.  Since you've seen code to perform many of the tasks in previous labs, many of the cells will be hidden.  If it would be helpful to inspect the code, you can always click the eye icon to the left of a cell.
"""

# ╔═╡ 2e51744b-b040-4f21-94b8-ffe9cd1e149e
begin
	fn = joinpath("../_assets/week4/legacy_data.csv")
	if !isfile(fn) || !(filesize(fn)>0)
		path = joinpath(pwd(),"data")
		mkpath(path)
		fn = joinpath(path,"legacy_data.csv")
		fn = Downloads.download("https://github.com/leerosenthalj/CLSI/raw/master/legacy_tables/legacy_data.csv", fn)
	end
	if filesize(fn) > 0
		df_all = CSV.read(fn, DataFrame)
		select!(df_all,["name","jd","mnvel","errvel", "cts","sval","tel"])
		# Rename columns to match labels from table in original paper
		rename!(df_all, "name"=>"Name","jd"=>"d","mnvel"=>"RVel","errvel"=>"e_RVel","tel"=>"Inst", "sval"=>"SVal")	
		star_names = unique(df_all.Name)
		md"Successfully, read RVs for $(length(star_names)) stars from California Legacy Survey from [https://github.com/leerosenthalj/CLSI](https://github.com/leerosenthalj/CLSI) & from [Rosenthal et al. (2021)](https://doi.org/10.3847/1538-4365/abe23c) into `df_all`."
	else
		df_all = DataFrame()
		star_names = String[""]

		danger(md"Error reading data file with RVs.  Expect empty plots below.")
	end
	
end

# ╔═╡ d79fc353-e30e-49ab-aa8e-9ba4b76a879b
md"""
Select star to analyze.
"""

# ╔═╡ 9404128d-0638-45ba-aaf6-a6ea47489b49
star_name_to_plt = "217014"

# ╔═╡ 253decc1-35c7-4454-b500-4f28e1087d36
starid = searchsortedfirst(star_names,star_name_to_plt);

# ╔═╡ 5e92054a-ca9e-4949-9727-5a9ed14003c0
begin
	star_name = star_names[starid]
	df_star = df_all |> @filter( _.Name == star_name ) |> DataFrame
end;

# ╔═╡ ffd80564-ea2a-40c3-8250-5c9482ab641d
md"""
Group data according to which instrument made the observation, so we can analyze each dataset separately.
"""

# ╔═╡ bce3f35c-07a1-48ef-8a29-243b2215fcb5
begin 
	df_star_by_inst = DataFrame()
	try
	df_star_by_inst = df_star |> @groupby( _.Inst ) |> @map( {bjd = _.d, rv = _.RVel, σrv = _.e_RVel, inst= key(_), nobs_inst=length(_) }) |> DataFrame;
	catch
	end
end;

# ╔═╡ 174f6c6d-6ff9-449d-86b3-85bcad9f01a2
 begin  # Make more useful observatory/instrument labels
	instrument_label = Dict(zip(["j","k","apf","lick"],["Keck (post)","Keck (pre)","APF","Lick"]))
	for k in keys(instrument_label)  
		if k ∉ df_star_by_inst.inst
			delete!(instrument_label,k)
		end
	end
	instrument_label 
end;

# ╔═╡ 8b1f8b91-12b5-4e61-a8ff-63538189cf34
t_offset = 2455000;  # So labels on x-axis are more digestable

# ╔═╡ 5edc2a2d-6f63-4ac6-8c33-2c5d670bc466
begin
	plt_rv_all_inst = plot() #legend=:none, widen=true)
	local num_inst = size(df_star_by_inst,1)
	for inst in 1:num_inst
		rvoffset = mean(df_star_by_inst[inst,:rv])
		scatter!(plt_rv_all_inst,df_star_by_inst[inst,:bjd].-t_offset,
				df_star_by_inst[inst,:rv].-rvoffset,
				yerr=collect(df_star_by_inst[inst,:σrv]),
				label=instrument_label[df_star_by_inst[inst,:inst]], markercolor=inst)
				#markersize=4*upscale, legendfontsize=upscale*12
	end
	xlabel!(plt_rv_all_inst,"Time (d)")
	ylabel!(plt_rv_all_inst,"RV (m/s)")
	title!(plt_rv_all_inst,"HD " * star_name )
	plt_rv_all_inst
end

# ╔═╡ 811ed6ac-4cf2-435a-934c-edfbb38564b2
begin
	select_obs_cell_id = PlutoRunner.currently_running_cell_id[] |> string
	select_obs_cell_url = "#$(select_obs_cell_id)"
	md"""
Select which instrument's data to analyze below: $(@bind inst_to_plt Select(collect(values(instrument_label)); default="Keck (post)"))
"""
end

# ╔═╡ b53fc91a-d2ed-4727-a683-205092e33bc6
inst_idx = findfirst(isequal(inst_to_plt),map(k->instrument_label[k], df_star_by_inst[:,:inst]));

# ╔═╡ 10f95d69-9cd8-47d4-a534-8de09ea3b216
begin
	plt_1inst = plot(xlabel="Time (d)", ylabel="RV (m/s)", title="Zoom in on RVs from instrument being fit") 
	scatter!(plt_1inst, df_star_by_inst[inst_idx,:bjd].-t_offset,df_star_by_inst[inst_idx,:rv],yerr=df_star_by_inst[inst_idx,:σrv], label=instrument_label[df_star_by_inst[inst_idx,:inst]], markercolor=inst_idx)
end

# ╔═╡ b821a2ae-bf16-4018-b85a-ff1713f40103
md"""
**Q1a:** Which observatory has provided the most observations to this dataset?  
"""

# ╔═╡ 2f09622b-838c-42df-a74f-81960916fae2
response_1a = missing

# ╔═╡ 21834080-14de-4926-9766-5a3ad994e2a1
md"""
# Fitting RV Model to data
"""

# ╔═╡ 3c14ec5c-e72a-4af4-8859-fd7a0bf91409
md"""
Make NamedTuple with time, observed rv and measurment uncertainties for data from instrument to be analyzed below.
"""

# ╔═╡ 5d61ebb8-465a-4d10-a0e6-a0c043f511b5
md"""
Set initial guesses for model parameters.
"""

# ╔═╡ 99e98b44-1994-4d0f-ba38-f10887a1be0c
md"""
Fit circular model using general linear regression to get good guesses for the orbital amplitude and phase.
"""

# ╔═╡ 49fdca20-46fd-4f31-94f1-ed58f3b32305
md"""
Fit 1-planet model: $(@bind try_fit_1pl CheckBox())
"""

# ╔═╡ 29352332-7fae-4709-9883-cfb480650a6c
md"""
Plotting the RV versus phase (rather than time) makes it much easier to see the orbit of the planet when the time span of observations is much longer than the orbital period of the planet.
Therefore, we'll use the best-fit orbital period fit above to compute the orbital phase of the planet at each time.  
We'll apply an arbitrary vertical offset to observations from each observatory, so you can compare the data from the different observatories.
"""

# ╔═╡ 0a508687-c2b7-466d-a5d6-2d1792687f3a
md"""
Since the planet causes large amplitude RV variations, it's useful to look at the residuals between the observations and the best-fit model.  We can plot residual RV versus either time or orbital phase.
"""

# ╔═╡ cf582fc9-07b1-4d09-b379-2576924c026b
md"""
**Q1b:** What is the typical scatter of RV residuals about the best-fit model for data from each observatory?  What observatory has the smallest scatter in the residuals between the observations and the best-fit model?  
"""

# ╔═╡ 9007735e-45c2-4a96-8ea5-03d7b8b58410
response_1b = missing

# ╔═╡ ba869a69-167e-4a1c-92af-e8592f6fca3d
md"""
**Q1c:** For which observatory do you anticipate that the dervied radial velocity amplitude ($K$) will have the smallest uncertainties?  
"""

# ╔═╡ 220caa90-90e8-4a52-a133-e37bb9cf5b50
response_1c = missing

# ╔═╡ 11469768-34af-470a-b431-c47b17d6a586
Markdown.parse("""
In the next sections, we'll analyze the data from one observatory at a time (using the [drop-down box above]($(select_obs_cell_url))).  I'd suggest starting with Keck.  Then, you can repeat the analysis using data from APF (or Lick observatory).  
""")

# ╔═╡ 710af3aa-b842-43b2-ab96-cda80b2a2ee0
md"""
## Bootstrap Estimate of Uncertainties
"""

# ╔═╡ b89645c8-6574-4f53-b40e-5c4e4236671e
md"""
Number of Bootstrap Samples: $(@bind num_bootstrap_samples NumberField(1:1000, default= 200))    
$(@bind redraw_bootstrap Button("Redraw bootstrap samples"))
"""

# ╔═╡ b9d0ecc6-f9a8-4107-9945-f17aa09e0b87
md"""
Run bootstrap analysis with 1-planet model: $(@bind try_bootstrap_1pl CheckBox())
"""

# ╔═╡ ebec340e-297f-44c1-8095-60ea68dd530c
md"""
Once you've selected the observatory/instrument whose data you want to analyze, check the box above to trigger the cell below to generate many synthetic datasets by draw "bootstrap" samples from the actual data (with replacement), to attempt to find the best-fit parameters for each synthetic dataset.  We can visualize the disitribution of the resulting fits to the bootstrap simulations to get an estimate of the uncertainties in the model parameters.  If you'd like to get smoother histograms below (and more precise estimates of the mean and standard deviation of the parameters), then you can boost the number of bootstrap samples.  
"""

# ╔═╡ 7296def6-31c0-4df2-b6ca-2fd953bdfb1f
md"""
### Cross-validation
We can compute the distribution of RMS of residuals between the RV measurements and the predictions from the best-fit model for each bootstrap sample.  We evaluate this separately for the points used for fitting the model and the points excluded from fitting the model.  
"""

# ╔═╡ 0737daef-b8f1-49ef-9a06-5cf1b716f719
response_2a = missing

# ╔═╡ 10a2b5b5-6a84-4b1e-a5f6-dd2434541edb
md"""
**Q2b:**  If you choose to analyze RVs from the Automated Planet Finder (APF) or Lick observatory, then how does the RMS of RV residuals change (relative to the RV residuals from the best-fit to data from Keck observatory)?  What could explain these differences?
"""

# ╔═╡ 390d9fc3-22f1-4e46-8164-4fc33f494035
response_2b = missing

# ╔═╡ 8743b110-ed40-4718-8fc3-e296ee8339f2
md"""
## Markov chain Monte Carlo Estimate of Uncertainties
"""

# ╔═╡ 67f3aef3-f34f-4e67-8ff4-adb8aa0284db
md"""
While the bootstrap method has some intuitive appeal, the theoretical basis is not as strong as performing proper Bayesian inference.  Therefore, you'll compare the uncertainty estimates from the two methods below.  
"""

# ╔═╡ 6a141962-d4d6-4f27-b94e-2d0aee0740c7
protip(md"""
There are some scenarios in which the bootstrap can significantly underestimate the true uncertainties (e.g., when the time span of observations is not much longer than the orbital period of the planet).  For this dataset an model, the bootstrap would be expected to give reasonable uncertainty estimates.
""")

# ╔═╡ 5c9f6b52-87d3-4971-90f4-3f953f7bce7f
md"""
Near the bottom of the notebook, there is code to fit a [Keplerian RV model](#69f40924-6b24-4014-8c1b-f600a0759aab) using Turing.jl.   Below, we'll define a posterior distribution conditioned on the actual data and choose the parameters to start the Marko chain simulations from.
"""

# ╔═╡ 1af9fbff-4786-415c-b1e6-ecc580e22ef2
md"""
The MCMC simulations below will take several minutes to run.  To prevent the notebook from locking up, I've set it so that it won't trigger new computations until you have checked the box below *and* you click the submit button.  If you want to disable the MCMC simulations (e.g., so you can experiment with bootstrap simulations for different observatories above), then uncheck the box below and click the submit button, so that the notebook know not to run the MCMC simulations (until you recheck the box and click submit).
"""

# ╔═╡ fc8d934c-7557-4011-a1e1-961bfd4e11a1
@bind mcmc confirm(PlutoUI.combine() do Child
md"""
**Markov chain Monte Carlo Parameters**

Steps per chain  $(Child("num_steps_per_chain", NumberField(100:100:10_000; default=100))) 
Burn-in steps per chain  $(Child("num_steps_burn_in_per_chain", NumberField(100:100:10_000; default=200))) 

Number of chains  $(Child("num_chains", NumberField(1:10; default=4)))  
Ready to run MCMC simulations: $(Child("run", CheckBox()))
""" end )

# ╔═╡ 86aa15f3-b525-4c77-ab01-159b8e977314
aside(tip(md"""You will see lots of warning messages about "The current propsoal will be rejected due to numerical error(s).".  That's because our model has some hard boundaries (e.g., eccentricity can not exceed 1, K must be positive).  When the Markov chain proposes a state outside the support of the priors, the model returns -Inf for the log probability.  This prevents the Markov chain from accepting invalid proposals, but creates lots of warning messages."""), v_offset=-100)

# ╔═╡ 962d20ef-8446-4894-80df-725c1bac04be
md"""
It's always good to check trace plots to see if there's anything suspicious in the Markov chains.
"""

# ╔═╡ 300a1ea8-023e-4c38-8a7e-1bc9ac62c311
md"""
Next, we'll inspect the distributions of the marginal posterior for several of the model parameters.  
"""

# ╔═╡ 7e1a46fd-392c-412c-8a5c-e54765112564
md"""
**Q3a:** Do you notice significant differences in the traces of model parameters from the different Markov chains (in different colors)?  Or in the histograms of marginal posterior distributions?  If so, consider running the Markov chains longer.  (But if the run time starts to exceed ~5 minutes, then do the best you can with the results you can get quickly.  You're welcome to run much longer chains (e.g., while you take a break or go eat), but that's not required.)
"""

# ╔═╡ a935174a-1057-4ad6-9b92-84981f4a4bb2
response_3a = missing

# ╔═╡ eb96986f-78fe-4a28-9ccf-6d3a66f063a6
md"""
We can compute summary statistics (e.g., sample mean, sample standard deviation, various quantiles)  for each model parameter from each Markov chain.  
"""

# ╔═╡ 3cf57331-688b-4b71-83f1-51cf53cfb0ee
md"Chain to calculate summary statistics for: $(@bind chain_id Slider(1:mcmc.num_chains;default=1))"

# ╔═╡ 7a2e757b-4117-455d-ba41-6205ec4746dd
md"**Summary statistics for chain $chain_id**"

# ╔═╡ 121869f2-c78d-4b46-bd5d-9d97a2f68e54
md"""
**Q3b:** Compare the mean K and sample standard deviation of K from each of the Markov chains you computed. How does the dispersion of K values within each chain compare to the dispersion of K values across all the chains?  What does this imply for the uncertainty in K?
"""

# ╔═╡ 96fc2d52-5128-483c-9962-817f1b013065
response_3b = missing

# ╔═╡ abad7a4d-0bb8-4c8f-bdec-f9e0d2839fd7
md"""
Now, repeat the analysis for data from another instrument.  (Analyzing the APF data will be a little faster, since Lick has more data points.)

**Q3d:** How do the estimates for the mean and uncertainty for K compare across the two observatories?
"""

# ╔═╡ afadd762-5eb8-47ca-82b3-0862299e5fb9
response_3d = missing

# ╔═╡ e325a28f-c8ef-4f0c-8f29-e9f4c34ea746
md"""
**Q3e:** Based on the combination of results, what do you conclude for the accuracy of the measurement of K?
"""

# ╔═╡ 244eccc2-463d-453b-bc30-1decbf0eed9a
response_3e = missing

# ╔═╡ bb1e1664-0c67-4aea-9e76-37669d253592
md"""
**Q3f:** How did the results compare to your predictions in Q1c?  If you found results different than you expected, try to explain the reason for the difference?
"""

# ╔═╡ 66f4acf5-152c-4792-9d7f-9a0ddb6459f6
response_3f = missing

# ╔═╡ b60aadbc-4e70-414e-9fdc-c3b042cb17bf
md"# Setup"

# ╔═╡ 69f40924-6b24-4014-8c1b-f600a0759aab
md"## Keplerian Radial Velocity Model"

# ╔═╡ 5f438b58-2f87-4373-b9ca-e35673b7b46f
@model rv_kepler_model_v1(t, rv_obs, σ_obs) = begin
    # Specify Priors
	P_max = 1000
	K_max = 1000
	σj_max = 100
    P ~ ModifiedJeffreysPriorForScale(1.0, P_max)        # orbital period
    K ~ ModifiedJeffreysPriorForScale(1.0, K_max)        # RV amplitude
    #e ~ Truncated(Rayleigh(0.3),0.0,0.999);              # orbital eccentricity
    #ω ~ Uniform(0, 2π)           # arguement of pericenter
	h ~ Normal(0,0.3)
	k ~ Normal(0,0.3)
    M0_minus_ω ~ Uniform(0,2π)   # mean anomaly at t=0 minus ω
    C ~ Normal(0,1000.0)         # velocity offset
    #σ_j ~ ModifiedJeffreysPriorForScale(1.0, σj_max)      # magnitude of RV jitter
	σ_j ~ LogNormal(log(1.0), 0.5)      # magnitude of RV jitter
        
    # Transformations to make sampling more efficient
	e = sqrt(h^2+k^2)
	ω = atan(h,k)
    M0 = M0_minus_ω + ω

    # Reject any parameter values that are unphysical, _before_ trying 
    # to calculate the likelihood to avoid errors/assertions
    if !(0.0 <= e < 1.0)      
        Turing.@addlogprob! -Inf
        return
    end

    # Likelihood
    # Calculate the true velocity given model parameters
    rv_true = calc_rv_keplerian_plus_const.(t, P,K,e,ω,M0,C)
        
    # Specify measurement model
    σ_eff = sqrt.(σ_obs.^2 .+ σ_j.^2)
    rv_obs ~ MvNormal(rv_true, σ_eff )
end

# ╔═╡ a7514405-af4c-4f16-8508-91ee624d8a1c
function calc_true_anom(ecc_anom::Real, e::Real)
	true_anom = 2*atan(sqrt((1+e)/(1-e))*tan(ecc_anom/2))
end

# ╔═╡ 690205fb-0b95-4614-9b66-dec362ed693c
begin
	calc_ecc_anom_cell_id = PlutoRunner.currently_running_cell_id[] |> string
	calc_ecc_anom_url = "#$(calc_ecc_anom_cell_id)"
	"""
	   `calc_ecc_anom( mean_anomaly, eccentricity )`
	   `calc_ecc_anom( param::Vector )`
	
	Estimates eccentric anomaly for given 'mean_anomaly' and 'eccentricity'.
	If passed a parameter vector, param[1] = mean_anomaly and param[2] = eccentricity. 
	
	Optional parameter `tol` specifies tolerance (default 1e-8)
	"""
	function calc_ecc_anom end
	
	function calc_ecc_anom(mean_anom::Real, ecc::Real; tol::Real = 1.0e-8)
	  	if !(0 <= ecc <= 1.0)
			println("mean_anom = ",mean_anom,"  ecc = ",ecc)
		end
		@assert 0 <= ecc <= 1.0
		@assert 1e-16 <= tol < 1
	  	M = rem2pi(mean_anom,RoundNearest)
	    E = ecc_anom_init_guess_danby(M,ecc)
		local E_old
	    max_its_laguerre = 200
	    for i in 1:max_its_laguerre
	       E_old = E
	       E = update_ecc_anom_laguerre(E_old, M, ecc)
	       if abs(E-E_old) < tol break end
	    end
	    return E
	end
	
	function calc_ecc_anom(param::Vector; tol::Real = 1.0e-8)
		@assert length(param) == 2
		calc_ecc_anom(param[1], param[2], tol=tol)
	end;
end

# ╔═╡ 6c01ab20-217f-4671-9ade-8ac928a65771
Markdown.parse("""
There is no closed form solution for \$E\$, so we must solve for \$E(t)\$ iteratively 
(see [code for `calc_ecc_anom(M, e)`]($(calc_ecc_anom_url)).
""")

# ╔═╡ 3fbcc50d-9f6a-4aec-9a8f-f2f525223f0e
begin 
	""" Calculate RV from t, P, K, e, ω and M0	"""
	function calc_rv_keplerian end 
	calc_rv_keplerian(t, p::Vector) = calc_rv_keplerian(t, p...)
	function calc_rv_keplerian(t, P,K,e,ω,M0) 
		mean_anom = t*2π/P-M0
		ecc_anom = calc_ecc_anom(mean_anom,e)
		true_anom = calc_true_anom(ecc_anom,e)
		rv = K/sqrt((1-e)*(1+e))*(cos(ω+true_anom)+e*cos(ω))
	end
end

# ╔═╡ bab9033c-b9ee-45c1-9466-838e40bdb920
function make_rv_vs_phase_panel(e, ω; P::Real=1, K::Real=1, M0::Real =0, panel_label="", t_max::Real = P, xticks=false, yticks=false )
	plt = plot(legend=:none, widen=false, xticks=xticks, yticks=yticks, margin=0mm, link=:none)
	t_plt = collect(range(0,stop=t_max,length=1000))
	rv_plt = calc_rv_keplerian.(t_plt, P, K, e, ω, M0)
	plot!(plt,t_plt,rv_plt, linecolor=:black) #, lineweight=4)
	xlims!(0,P)
	if length(panel_label)>0
		annotate!(plt, 0.505,2.1, text(panel_label,24,:center))
	end
	return plt
end

# ╔═╡ ee7aaab9-5e4f-46ab-8100-75be142fba72
begin 
	plt_1pl = make_rv_vs_phase_panel(e_plt, ω_plt, P=P_plt, K=K_plt, M0=M0_plt, t_max = 100, xticks=true, yticks=true)
	xlabel!(plt_1pl, L"\mathrm{Time} \;\;\; (day)")
	ylabel!(plt_1pl, L"\Delta RV\;\;\;\; (m/s)")
	xlims!(plt_1pl,0,100)
	#ylims!(plt_1pl,-6,6)
end

# ╔═╡ cc7006c7-e3ef-470a-b93e-5743a27a32d9
begin 
	""" Calculate RV from t, P, K, e, ω, M0	and C"""
	function calc_rv_keplerian_plus_const end 
	calc_rv_keplerian_plus_const(t, p::Vector) = calc_rv_keplerian_plus_const(t, p...)
	
	function calc_rv_keplerian_plus_const(t, P,K,e,ω,M0,C) 
		calc_rv_keplerian(t, P,K,e,ω,M0) + C
	end
end

# ╔═╡ 4f047081-a4d6-414b-9c3e-0eb055c730b3
"""
   `ecc_anom_init_guess_danby(mean_anomaly, eccentricity)`

Returns initial guess for the eccentric anomaly for use by itterative solvers of Kepler's equation for bound orbits.  

Based on "The Solution of Kepler's Equations - Part Three"
Danby, J. M. A. (1987) Journal: Celestial Mechanics, Volume 40, Issue 3-4, pp. 303-312 (1987CeMec..40..303D)
"""
function ecc_anom_init_guess_danby(M::Real, ecc::Real)
	@assert -2π<= M <= 2π
	@assert 0 <= ecc <= 1.0
    if  M < zero(M)
		M += 2π
	end
    E = (M<π) ? M + 0.85*ecc : M - 0.85*ecc
end;

# ╔═╡ 8f700e72-df0f-4e68-85fe-7fbe8da7fbb1
"""
   `update_ecc_anom_laguerre(eccentric_anomaly_guess, mean_anomaly, eccentricity)`

Update the current guess for solution to Kepler's equation
  
Based on "An Improved Algorithm due to Laguerre for the Solution of Kepler's Equation"
   Conway, B. A.  (1986) Celestial Mechanics, Volume 39, Issue 2, pp.199-211 (1986CeMec..39..199C)
"""
function update_ecc_anom_laguerre(E::Real, M::Real, ecc::Real)
  #es = ecc*sin(E)
  #ec = ecc*cos(E)
  (es, ec) = ecc .* sincos(E)  # Does combining them provide any speed benefit?
  F = (E-es)-M
  Fp = one(M)-ec
  Fpp = es
  n = 5
  root = sqrt(abs((n-1)*((n-1)*Fp*Fp-n*F*Fpp)))
  denom = Fp>zero(E) ? Fp+root : Fp-root
  return E-n*F/denom
end;

# ╔═╡ 7047d464-efdd-4315-b930-5b2e8a3d93c5
md"""
### Fitting Keplerian RV model
"""

# ╔═╡ e33e5fdc-7f14-4469-b462-6766ad3ce230
function find_best_1pl_fit(θinit::AbstractVector, loss::Function; num_init_phases::Integer=1, num_init_ωs::Integer=1, f_abstol::Real = 1e-2 )
	@assert 1 <= num_init_phases <= 32
	@assert 1 <= num_init_ωs <= 8
	result_list = Array{Any}(undef,num_init_phases, num_init_ωs)
	θinit_list = fill(θinit,num_init_phases, num_init_ωs)
	e_base = sqrt(θinit[3]^2+θinit[4]^2)
	ω_base = atan(θinit[3],θinit[4])
	for i in 1:num_init_phases 
		for j in 1:num_init_ωs
		Δω = (j-1)/num_init_ωs * 2π
		θinit_list[i,j][3] = e_base*sin(ω_base + Δω)
		θinit_list[i,j][4] = e_base*cos(ω_base + Δω)
		θinit_list[i,j][5] += (i-1)/num_init_phases * 2π - Δω
		θinit_list[i,j][5] = mod(θinit_list[i,j][5],2π)
		try 
			result_list[i,j] = Optim.optimize(loss, θinit_list[i,j], BFGS(), autodiff=:forward, Optim.Options(f_abstol=f_abstol));
		catch
			result_list[i,j] = (;minimum=Inf)
		end
		end
	end
	best_result_id = argmin(map(r->r.minimum, result_list))
	result = result_list[best_result_id]
end

# ╔═╡ 56d09fea-e2c2-4345-a089-419ac863ac43
""" Calculate RV from t, P, K, e, ω, M0	and C with optional slope and t_mean"""
function model_1pl(t, P, K, e, ω, M, C; slope=0.0, t_mean = 0.0)
	calc_rv_keplerian(t-t_mean,P,K,e,ω,M) + C + slope * (t-t_mean)
end

# ╔═╡ 3932fb9d-2897-4d64-8dba-a51799d1aa7a
""" Convert vector of (P,K,h,k,M0-ω) to vector of (P, K, e, ω, M0) """
function PKhkωMmω_to_PKeωM(x::Vector) 
	(P, K, h, k, ωmM) = x
	e = sqrt(h^2+k^2)
	ω = atan(h,k)
	return [P, K, e, ω, ωmM+ω]
end

# ╔═╡ d4e5cd92-21c0-4073-ab3e-8cd5804976c8
md"""
### Loss function
"""

# ╔═╡ 71a560d1-efb4-48ad-8a45-ac4fd64537fd
function make_loss_1pl(data; t_mean=0)
	function loss_1pl(θ) 
		(P1, K1, h1, k1, Mpω1, C, σj ) = θ
		( P1, K1, e1, ω1, M1 ) = PKhkωMmω_to_PKeωM([P1, K1, h1, k1, Mpω1])
		if e1>1 return 1e6*e1 end
		rv_model = model_1pl.(data.t,P1,K1,e1,ω1,M1,C, t_mean=t_mean)
		loss = 0.5*sum( (rv_model.-data.rv).^2 ./ (data.σrv.^2 .+ σj^2) )
		loss += 0.5*sum(log.(2π*(data.σrv.^2 .+σj^2)))
		return loss
	end
end

# ╔═╡ 9283ac8f-c45e-42e0-ac41-579a3dd35e96
function make_neg_logp(
    model::Turing.Model,
    sampler=Turing.SampleFromPrior(),
    ctx::Turing.DynamicPPL.AbstractContext = DynamicPPL.DefaultContext()
)
    vi = Turing.VarInfo(model)

    # define function to compute log joint.
    function ℓ(θ)
        new_vi = Turing.VarInfo(vi, sampler, θ)
        model(new_vi, sampler, ctx)
        logp = Turing.getlogp(new_vi)
        return -logp
    end

end

# ╔═╡ 14cca39b-cded-4108-877f-842c70ff8843
md"### Generalized Linear Least Squares Fitting"

# ╔═╡ 50bdc155-4e41-4f8f-8b6e-12599138ec80
function fit_general_linear_least_squares( design_mat::ADM, covar_mat::APD, obs::AA ) where { ADM<:AbstractMatrix, APD<:AbstractPDMat, AA<:AbstractArray }
   Xt_inv_covar_X = Xt_invA_X(covar_mat,design_mat)
   X_inv_covar_y =  design_mat' * (covar_mat \ obs)
   AB_hat = Xt_inv_covar_X \ X_inv_covar_y                   # standard GLS
end

# ╔═╡ 65c6dea3-abe1-4226-a943-02d5a3138873
function calc_design_matrix_circ!(result::AM, period, times::AV) where { R1<:Real, AM<:AbstractMatrix{R1}, AV<:AbstractVector{R1} }
        n = length(times)
        @assert size(result) == (n, 2)
        for i in 1:n
                ( result[i,1], result[i,2] ) = sincos(2π/period .* times[i])
        end
        return result
end

# ╔═╡ 91d39e07-5507-41e2-89b4-4d5236f70cf8
function calc_design_matrix_circ(period, times::AV) where { R1<:Real, AV<:AbstractVector{R1} }
        n = length(times)
        dm = zeros(n,2)
        calc_design_matrix_circ!(dm,period,times)
        return dm
end

# ╔═╡ fd0f40ee-1e1a-4a04-aa89-dbe30cd387b3
md"""
### Custom prior distributions
"""

# ╔═╡ fcb2cae4-5f57-4c77-a463-3112a2853280
begin
        struct ModifiedJeffreysPriorForScale{T1,T2,T3} <: ContinuousUnivariateDistribution where { T1, T2, T3 }
                scale::T1
                max::T2
                norm::T3
        end
        
        function ModifiedJeffreysPriorForScale(s::T1, m::T2) where { T1, T2 }
                @assert zero(s) < s && !isinf(s)
                @assert zero(m) < m && !isinf(s)
                norm = 1/log1p(m/s)         # Ensure proper normalization
                ModifiedJeffreysPriorForScale{T1,T2,typeof(norm)}(s,m,norm)
        end
        
        function Distributions.rand(rng::AbstractRNG, d::ModifiedJeffreysPriorForScale{T1,T2,T3}) where {T1,T2,T3}
                u = rand(rng)               # sample in [0, 1]
                d.scale*(exp(u/d.norm)-1)   # inverse CDF method for sampling
        end

        function Distributions.logpdf(d::ModifiedJeffreysPriorForScale{T1,T2,T3}, x::Real) where {T1,T2,T3}
                log(d.norm/(1+x/d.scale))
        end
        
        function Distributions.logpdf(d::ModifiedJeffreysPriorForScale{T1,T2,T3}, x::AbstractVector{<:Real})  where {T1,T2,T3}
            output = zeros(x)
                for (i,z) in enumerate(x)
                        output[i] = logpdf(d,z)
                end
                return output
        end
        
        Distributions.minimum(d::ModifiedJeffreysPriorForScale{T1,T2,T3})  where {T1,T2,T3} = zero(T2)
        Distributions.maximum(d::ModifiedJeffreysPriorForScale{T1,T2,T3})  where {T1,T2,T3} = d.max
        
        custom_prob_dist_url = "#" * (PlutoRunner.currently_running_cell_id[] |> string)
	ModifiedJeffreysPriorForScale
end

# ╔═╡ fcf19e04-3e35-4a01-8036-fd5b283fdd37
if size(df_star_by_inst,1)>0  # Warning: Picks RVs from only 1 instrument
	data = ( t=collect(df_star_by_inst[inst_idx,:bjd]).-t_offset,
			rv=collect(df_star_by_inst[inst_idx,:rv]),
			σrv=collect(df_star_by_inst[inst_idx,:σrv]) )
	t_mean = mean(data.t)
	t_plt = range(minimum(data.t), stop=maximum(data.t), step=1.0)
	phase_plt = range(0.0, stop=1, length=100)
else
	data = (t=Float64[], rv=Float64[], σrv=Float64[])
end;

# ╔═╡ 44eb9ddd-e04f-4961-b11e-50df17731516
begin
	P_guess = 4.230785  # for 51 Peg b
	h_guess = 0.01      
	k_guess = 0.01
	C_guess = mean(data.rv)
	σj_guess = 3.0
end;

# ╔═╡ eb83fe44-d67d-43cd-b823-b29b965a17af
begin
	param_fit_linear = fit_general_linear_least_squares( 
       calc_design_matrix_circ(P_guess,data.t), PDiagMat(data.σrv), data.rv)
	K_guess = sqrt(param_fit_linear[1]^2+param_fit_linear[2]^2)
	phase_guess = atan(param_fit_linear[1],param_fit_linear[2])
	θinit1 = [P_guess, K_guess, h_guess, k_guess, mod(phase_guess-atan(h_guess, k_guess),2π), C_guess, σj_guess] 
end

# ╔═╡ 41b2eea0-3049-4fa5-803e-83a54b74ef27
if try_fit_1pl && @isdefined data 
	loss_1pl_all_data = make_loss_1pl(data, t_mean=t_mean)
	result = find_best_1pl_fit(θinit1, loss_1pl_all_data, num_init_phases = 1, num_init_ωs=4)
end

# ╔═╡ 82e06e8b-8211-48bc-a719-743a616a2c8b
if @isdefined result
	period_to_phase_by = result.minimizer[1]
end

# ╔═╡ 43ae8d15-6381-4c86-b08d-2d12cd4bc653
if @isdefined result
	#upscale
	plt_phase_all = plot(widen=false)
	num_inst = size(df_star_by_inst,1)
	for inst in 1:num_inst
		if length(df_star_by_inst[inst,:rv]) == 0 continue end
		rvoffset = mean(df_star_by_inst[inst,:rv]) .- 30 .* (inst-2)
		phase = mod.((df_star_by_inst[inst,:bjd].-t_offset)./period_to_phase_by,1.0)
		scatter!(plt_phase_all,phase, 
				df_star_by_inst[inst,:rv].-rvoffset,
				yerr=collect(df_star_by_inst[inst,:σrv]),
				markersize=2, markerstrokewidth=0.5,
				label=instrument_label[df_star_by_inst[inst,:inst]])
	end
	#plot!(plt,t_plt,pred_1pl, label=:none)
	xlabel!(plt_phase_all,"Phase")
	ylabel!(plt_phase_all,"RV (m/s)")
	title!(plt_phase_all,"HD " * star_name )
	plt_phase_all
end

# ╔═╡ 1e63307b-bb3b-4b32-9983-90626e1a3dde
if @isdefined result
	θinit_mcmc = copy(result.minimizer)
	# Make sure initial state has non-zero prior probability
	θinit_mcmc[5] = mod(θinit_mcmc[5],2π)
end;

# ╔═╡ abc38d23-8665-4377-9a25-9e9c5a10a7bf
if @isdefined result 
	model_resid = data.rv.-
				model_1pl.(data.t,PKhkωMmω_to_PKeωM(result.minimizer[1:5])...,result.minimizer[6], t_mean=t_mean)
end;

# ╔═╡ 844ede38-9596-47a6-b30b-9eff622a2330
if @isdefined model_resid
	plt_resid = plot(legend=:none, widen=true)
	scatter!(plt_resid,data.t,
				model_resid,
				yerr=data.σrv, markercolor=inst_idx) 
	xlabel!(plt_resid,"Time (d)")
	ylabel!(plt_resid,"RV (m/s)")
	
	plt_resid_phase = plot(legend=:none, widen=false)
	phase = mod.(data.t ./ period_to_phase_by,1.0)
	scatter!(plt_resid_phase,phase,
				model_resid,
				yerr=data.σrv, markercolor=inst_idx) 
	xlabel!(plt_resid_phase,"Phase")
	ylabel!(plt_resid_phase,"RV (m/s)")
	title!(plt_resid,"HD " * star_name * " (residuals to 1 planet model)")
	plot(plt_resid, plt_resid_phase, layout=(2,1) )
end

# ╔═╡ bed8ac6f-052a-4cb3-9fb8-5162f1683dd2
if try_bootstrap_1pl 
	redraw_bootstrap
	results_bootstrap = Array{Any}(undef,num_bootstrap_samples)
	rms_bootstrap = zeros(num_bootstrap_samples)
	rms_bootstrap_train = zeros(num_bootstrap_samples)
	for i in 1:num_bootstrap_samples
		# Select sample of points to fit
		idx = sample(1:length(data.t),length(data.t))
		# Create NamedTuple with view of resampled data
		data_tmp = (;t=view(data.t,idx), rv=view(data.rv,idx), σrv=view(data.σrv,idx))
		loss_tmp = make_loss_1pl(data_tmp, t_mean=t_mean)
		# Attempt to find best-fit parameters results for resampled data
		results_bootstrap[i] = find_best_1pl_fit(result.minimizer, loss_tmp, num_init_phases=1, num_init_ωs=4)
		# Evaluate residuals for cross validation 
		if hasfield(typeof(results_bootstrap[i]),:minimizer)
			idx_test = filter(i->!(i∈idx), 1:length(data.t))
			pred_1pl = map(t->model_1pl(t,PKhkωMmω_to_PKeωM(results_bootstrap[i].minimizer[1:5])...,results_bootstrap[i].minimizer[6], t_mean=t_mean),view(data.t,idx_test))
			resid = view(data.rv,idx_test).-pred_1pl			
			rms_bootstrap[i] = sqrt(mean(resid.^2))
			idx_train = filter(i->(i∈idx), 1:length(data.t))
			pred_1pl = map(t->model_1pl(t,PKhkωMmω_to_PKeωM(results_bootstrap[i].minimizer[1:5])...,results_bootstrap[i].minimizer[6], t_mean=t_mean),view(data.t,idx_train))
			resid = view(data.rv,idx_train).-pred_1pl			
			rms_bootstrap_train[i] = sqrt(mean(resid.^2))
		end
	end
	results_bootstrap
end;

# ╔═╡ 4c890552-529c-46e1-bb1e-cbcea7f24672
if try_bootstrap_1pl 
	plt_bootstrap_resid = plot(xlabel="RMS RV Residuals (m/s)", ylabel="Samples")
	histogram!(plt_bootstrap_resid, rms_bootstrap, label="Test points", alpha=0.5)
	histogram!(plt_bootstrap_resid, rms_bootstrap_train, label="Training points", alpha=0.5)
end

# ╔═╡ 5d631862-97e7-4ccd-ab6f-a875989dde99
posterior_1 = rv_kepler_model_v1(data.t,data.rv,data.σrv);

# ╔═╡ 9371ce38-9cb5-4664-ae24-5554f4847868
if mcmc.run 
	if (Sys.iswindows() || (Threads.nthreads()==1))
        chains = sample(posterior_1, NUTS(), mcmc.num_steps_per_chain, discard_initial=mcmc.num_steps_burn_in_per_chain, init_params = θinit_mcmc)
	else
        chains = sample(posterior_1, NUTS(), MCMCThreads(), mcmc.num_steps_per_chain, mcmc.num_chains, discard_initial=mcmc.num_steps_burn_in_per_chain, init_params = fill(θinit_mcmc,mcmc.num_chains) )
	end
	summarystats(chains)
end

# ╔═╡ bd82489d-5156-4d88-98ea-8f05eab22e00
if mcmc.run
	summarystats(chains)
end

# ╔═╡ 6c2ab644-4799-4bf3-aac0-431bf399dab0
if mcmc.run
let 
	plt_title = plot(title = "MCMC Trace Plots", grid = false, showaxis = false, ticks=:none, bottom_margin = -25Plots.px)
    plt_P = traceplot(chains,:P, leftmargin=15Plots.px)
	ylabel!(plt_P, "P")
	title!(plt_P,"")
	plt_K = traceplot(chains,:K)
	ylabel!(plt_K, "K") 
	title!(plt_K,"")
    plt_e = traceplot(chains,:h, leftmargin=15Plots.px)
	ylabel!(plt_e, "h")
	title!(plt_e,"")
    plt_ω = traceplot(chains,:k)
	ylabel!(plt_ω, "k")
	title!(plt_ω,"")
    plt_ωpM = traceplot(chains,:M0_minus_ω, leftmargin=15Plots.px)
	ylabel!(plt_ωpM, "M₀-ω")
	title!(plt_ωpM, "")
	plt_σj = traceplot(chains,:σ_j)
	ylabel!(plt_σj, "σⱼ")
	title!(plt_σj, "")

    plot(plt_title,plt_P, plt_K, plt_e, plt_ω, plt_ωpM, plt_σj, layout=@layout([A{0.01h}; [B C; D E; F G]]), size=(600,800) )
end
end

# ╔═╡ 60a09c22-4100-41e9-8ecc-bfefde1492e0
if mcmc.run
	summarystats(chains[:,:,chain_id])
end

# ╔═╡ c7fe279b-d978-4b97-a031-bb933b64d90f
if mcmc.run
	quantile(chains[:,:,chain_id])
end

# ╔═╡ a9a0bf9f-4ab5-42c5-aa5e-24678ba5ca5a
if @isdefined results_bootstrap
	plt_title = plot(title = "Bootstrap Results", grid = false, showaxis = false, ticks=:none, bottom_margin = -25Plots.px)
	
	local Psample = map(r->r.minimizer[1],results_bootstrap)
	P_mean_bootstrap = mean(Psample)
	P_std_bootstrap = std(Psample)
	plt_P_hist = plot(xlabel="P (d)",ylabel="Samples",xticks=
	optimize_ticks(minimum(Psample),maximum(Psample),k_max=3)[1])
	histogram!(plt_P_hist,Psample, label=:none, nbins=50)
	
	local Ksample = map(r->r.minimizer[2],results_bootstrap)
	K_mean_bootstrap = mean(Ksample)
	K_std_bootstrap = std(Ksample)
	plt_K_hist = plot(xlabel="K (m/s)", ylabel="Samples",xticks=
	optimize_ticks(minimum(Ksample),maximum(Ksample),k_max=3)[1])
	histogram!(plt_K_hist,Ksample, label=:none, nbins=50)
	
	local esample = map(r->PKhkωMmω_to_PKeωM(r.minimizer)[3],results_bootstrap)
	e_mean_bootstrap = mean(esample)
	e_std_bootstrap = std(esample)
	plt_e_hist = plot(xlabel="e", ylabel="Samples",xticks=
	optimize_ticks(minimum(esample),maximum(esample),k_max=3)[1])
	histogram!(plt_e_hist,esample, label=:none, nbins=50)
	
	local ωsample = map(r->PKhkωMmω_to_PKeωM(r.minimizer)[4],results_bootstrap)
	ω_mean_bootstrap = mean(ωsample)
	ω_std_bootstrap = std(ωsample)
	plt_ω_hist = plot(xlabel="ω", ylabel="Samples",xticks=
	optimize_ticks(minimum(ωsample),maximum(ωsample),k_max=3)[1])
	histogram!(plt_ω_hist,ωsample, label=:none, nbins=50)

	h_mean_bootstrap = mean(esample.*sin.(ωsample))
	h_std_bootstrap = std(esample.*sin.(ωsample))
	k_mean_bootstrap = mean(esample.*cos.(ωsample))
	k_std_bootstrap = std(esample.*cos.(ωsample))
	
	local Csample = map(r->r.minimizer[6],results_bootstrap)
	C_mean_bootstrap = mean(Csample)
	C_std_bootstrap = std(Csample)
	plt_C_hist = plot(xlabel="C", ylabel="Samples",xticks=
	optimize_ticks(minimum(Csample),maximum(Csample),k_max=2)[1])
	histogram!(plt_C_hist,Csample, label=:none, nbins=50)
	
	local σjsample = map(r->r.minimizer[7],results_bootstrap)
	σj_mean_bootstrap = mean(σjsample)
	σj_std_bootstrap = std(σjsample)
	plt_σj_hist = plot(xlabel="σⱼ", ylabel="Samples",xticks=
	optimize_ticks(minimum(σjsample),maximum(σjsample),k_max=3)[1])
	histogram!(plt_σj_hist,σjsample, label=:none, nbins=50)
		
	plot(plt_title, plt_P_hist, plt_K_hist, plt_e_hist, plt_ω_hist, plt_C_hist, plt_σj_hist, layout=@layout([A{0.01h}; [B C; D E; F G ]]), size=(600,600) )
end

# ╔═╡ d305c716-080f-4faa-9143-bc75bfa6ce6f
if try_bootstrap_1pl 
	md"""
**Q2a:** Compare the distribution of the residuals for the training and test sets to:
1. the mean measurement uncertainty ($(round(mean(data.σrv),sigdigits=3)) m/s),
1. the jitter parameter fit to the data ($(round(σj_mean_bootstrap, sigdigits=3)) m/s), and
1. the quadrature sum of the two ($(round(sqrt(mean(data.σrv)^2+σj_mean_bootstrap^2),sigdigits=3)) m/s). 
"""
end

# ╔═╡ c7d3155c-5124-4724-bc7a-f1e2dcde879b
if try_bootstrap_1pl && (@isdefined chains)
md"""
**Q3c:** Compare the 
- mean K ($(round(mean(chains[:K]),sigdigits=4)) m/s) and 
- sample standard deviation of K ($(round(std(chains[:K]),sigdigits=4)) m/s)
estimated from the MCMC simulatiosn to the:
- mean K ($(round(K_mean_bootstrap,sigdigits=4)) m/s) and 
- sample standard deviation ($(round(K_std_bootstrap,sigdigits=4)) m/s)
estimated from the bootstrap simulations.  
Which method results in a larger estimated uncertainty?
"""
end

# ╔═╡ e2905cac-57f9-4da9-9160-828f46d72bb6
if mcmc.run
	let 
	plt_title = plot(title = "MCMC Posterior Samples", grid = false, showaxis = false, ticks=:none, bottom_margin = -25Plots.px)
    plt_P = histogram(get(chains,:P)[:P], nbins = 40, xticks=
	optimize_ticks(maximum(get(chains,:P)[:P]),minimum(get(chains,:P)[:P]),k_max=3)[1], label=:none, alpha=0.6, leftmargin=15Plots.px)
	xlabel!(plt_P, "P (d)") 
	ylabel!(plt_P, "")
	title!(plt_P,"")
	plt_K = histogram(chains,:K,nbins = 40 )
	xlabel!(plt_K, "K (m/s)") 
	title!(plt_K,"")
	chains_e = sqrt.(get(chains,:h)[:h].^2 .+get(chains,:k)[:k].^2)
	plt_e = histogram(chains_e, nbins = 40, xlabel="e", ylabel="", title="", leftmargin=15Plots.px, label=:none, alpha=0.6)
	chains_ω = atan.(get(chains,:h)[:h], get(chains,:k)[:k])
	xlabel!(plt_e, "e")
	title!(plt_e,"")
    plt_ω = histogram(chains_ω, nbins = 40, xlabel="ω", ylabel="", title="", label=:none, alpha=0.6)
    plt_ωpM = histogram(chains,:M0_minus_ω, nbins = 40, leftmargin=15Plots.px)
	xlabel!(plt_ωpM, "M₀-ω")
	ylabel!(plt_ωpM, "")
	title!(plt_ωpM, "")
	plt_σj = histogram(chains,:σ_j, nbins = 40)
	xlabel!(plt_σj, "σⱼ (m/s)")
	ylabel!(plt_σj, "")
	title!(plt_σj, "")

    plot(plt_title,plt_P, plt_K, plt_e, plt_ω, plt_ωpM, plt_σj, layout=@layout([A{0.01h}; [B C; D E; F G]]), size=(600,800) )
end
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
Downloads = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
Optim = "429524aa-4258-5aef-a3af-852621145aeb"
PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Query = "1a8c2f83-1ff3-5112-b086-8aa67b057ba1"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
Turing = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"

[compat]
CSV = "~0.10.3"
DataFrames = "~1.3.2"
Distributions = "~0.25.74"
ForwardDiff = "~0.10.32"
LaTeXStrings = "~1.3.0"
MCMCChains = "~5.4.0"
Optim = "~1.7.2"
PDMats = "~0.11.16"
Plots = "~1.26.0"
PlutoTeachingTools = "~0.1.7"
PlutoUI = "~0.7.37"
Query = "~1.0.0"
StatsBase = "~0.33.21"
Turing = "~0.21.12"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.0"
manifest_format = "2.0"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "69f7020bd72f069c219b5e8c236c1fa90d2cb409"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.2.1"

[[deps.AbstractMCMC]]
deps = ["BangBang", "ConsoleProgressMonitor", "Distributed", "Logging", "LoggingExtras", "ProgressLogging", "Random", "StatsBase", "TerminalLoggers", "Transducers"]
git-tree-sha1 = "5c26c7759412ffcaf0dd6e3172e55d783dd7610b"
uuid = "80f14c24-f653-4e6a-9b94-39d6b0f70001"
version = "4.1.3"

[[deps.AbstractPPL]]
deps = ["AbstractMCMC", "DensityInterface", "Setfield", "SparseArrays"]
git-tree-sha1 = "6320752437e9fbf49639a410017d862ad64415a5"
uuid = "7a57a42e-76ec-4ea3-a279-07e840d6d9cf"
version = "0.5.2"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.AbstractTrees]]
git-tree-sha1 = "5c0b629df8a5566a06f5fef5100b53ea56e465a0"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.2"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "195c5505521008abea5aee4f96930717958eac6f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.4.0"

[[deps.AdvancedHMC]]
deps = ["AbstractMCMC", "ArgCheck", "DocStringExtensions", "InplaceOps", "LinearAlgebra", "ProgressMeter", "Random", "Requires", "Setfield", "Statistics", "StatsBase", "StatsFuns", "UnPack"]
git-tree-sha1 = "0091e2e4d0a7125da0e3ad8c7dbff9171a921461"
uuid = "0bf59076-c3b1-5ca4-86bd-e02cd72cde3d"
version = "0.3.6"

[[deps.AdvancedMH]]
deps = ["AbstractMCMC", "Distributions", "Random", "Requires"]
git-tree-sha1 = "d7a7dabeaef34e5106cdf6c2ac956e9e3f97f666"
uuid = "5b7e9947-ddc0-4b3f-9b55-0d8042f74170"
version = "0.6.8"

[[deps.AdvancedPS]]
deps = ["AbstractMCMC", "Distributions", "Libtask", "Random", "StatsFuns"]
git-tree-sha1 = "9ff1247be1e2aa2e740e84e8c18652bd9d55df22"
uuid = "576499cb-2369-40b2-a588-c64705576edc"
version = "0.3.8"

[[deps.AdvancedVI]]
deps = ["Bijectors", "Distributions", "DistributionsAD", "DocStringExtensions", "ForwardDiff", "LinearAlgebra", "ProgressMeter", "Random", "Requires", "StatsBase", "StatsFuns", "Tracker"]
git-tree-sha1 = "e743af305716a527cdb3a67b31a33a7c3832c41f"
uuid = "b5ca4192-6429-45e5-a2d9-87aec30a685c"
version = "0.1.5"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.ArrayInterfaceCore]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "5bb0f8292405a516880a3809954cb832ae7a31c5"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.20"

[[deps.ArrayInterfaceStaticArraysCore]]
deps = ["Adapt", "ArrayInterfaceCore", "LinearAlgebra", "StaticArraysCore"]
git-tree-sha1 = "a1e2cf6ced6505cbad2490532388683f1e88c3ed"
uuid = "dd5226c6-a4d4-4bc7-8575-46859f9c95b9"
version = "0.1.0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "1dd4d9f5beebac0c03446918741b1a03dc5e5788"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.6"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "Future", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables", "ZygoteRules"]
git-tree-sha1 = "7fe6d92c4f281cf4ca6f2fba0ce7b299742da7ca"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.37"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.Bijectors]]
deps = ["ArgCheck", "ChainRulesCore", "ChangesOfVariables", "Compat", "Distributions", "Functors", "InverseFunctions", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "MappedArrays", "Random", "Reexport", "Requires", "Roots", "SparseArrays", "Statistics"]
git-tree-sha1 = "a3704b8e5170f9339dff4e6cb286ad49464d3646"
uuid = "76274a88-744f-5084-9051-94815aaf08c4"
version = "0.10.6"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings"]
git-tree-sha1 = "9310d9495c1eb2e4fa1955dd478660e2ecab1fbb"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.3"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "Statistics", "StructArrays"]
git-tree-sha1 = "a5fd229d3569a6600ae47abe8cd48cbeb972e173"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.44.6"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e7ff6cadf743c098e08fca25c91103ee4303c9bb"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.6"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "1833bda4a027f4b2a1c984baddcf755d77266818"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.1.0"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "1fd869cc3875b57347f7027521f561cf46d1fcd8"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.19.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "d08c20eef1f2cbc6e60fd3612ac4340b89fea322"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.9"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSolve]]
git-tree-sha1 = "332a332c97c7071600984b3c31d9067e1a4e6e25"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.1"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "78bee250c6826e1cf805a88b7f1e86025275d208"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.46.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.CompositionsBase]]
git-tree-sha1 = "455419f7e328a1a2493cabc6428d79e951349769"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.1"

[[deps.ConsoleProgressMonitor]]
deps = ["Logging", "ProgressMeter"]
git-tree-sha1 = "3ab7b2136722890b9af903859afcf457fa3059e8"
uuid = "88cd18e8-d9cc-4ea6-8889-5259c0d15c8b"
version = "0.1.2"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "fb21ddd70a051d882a1686a5a550990bbe371a95"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.4.1"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "1106fa7e1256b402a86a8e7b15c00c85036fef49"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.11.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "ae02104e835f219b8930c7664b8012c93475c340"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.2"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.DataValues]]
deps = ["DataValueInterfaces", "Dates"]
git-tree-sha1 = "d88a19299eba280a6d062e135a43f00323ae70bf"
uuid = "e7dc6d0d-1eca-5fa6-8ad6-5aecde8b7ea5"
version = "0.4.13"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "992a23afdb109d0d2f8802a30cf5ae4b1fe7ea68"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.11.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "70e9677e1195e7236763042194e3fbf147fdb146"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.74"

[[deps.DistributionsAD]]
deps = ["Adapt", "ChainRules", "ChainRulesCore", "Compat", "DiffRules", "Distributions", "FillArrays", "LinearAlgebra", "NaNMath", "PDMats", "Random", "Requires", "SpecialFunctions", "StaticArrays", "StatsBase", "StatsFuns", "ZygoteRules"]
git-tree-sha1 = "74dd5dac82812af7041ae322584d5c2181dead5c"
uuid = "ced4e74d-a319-5a8a-b0ac-84af2272839c"
version = "0.6.42"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.DynamicPPL]]
deps = ["AbstractMCMC", "AbstractPPL", "BangBang", "Bijectors", "ChainRulesCore", "ConstructionBase", "Distributions", "DocStringExtensions", "LinearAlgebra", "MacroTools", "OrderedCollections", "Random", "Setfield", "Test", "ZygoteRules"]
git-tree-sha1 = "7bc3920ba1e577ad3d7ebac75602ab42b557e28e"
uuid = "366bfd00-2699-11ea-058f-f148b4cae6d8"
version = "0.20.2"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EllipticalSliceSampling]]
deps = ["AbstractMCMC", "ArrayInterfaceCore", "Distributions", "Random", "Statistics"]
git-tree-sha1 = "4cda4527e990c0cc201286e0a0bfbbce00abcfc2"
uuid = "cad2338a-1db2-11e9-3401-43bc07c9ede2"
version = "1.0.0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.Extents]]
git-tree-sha1 = "5e1e4c53fa39afe63a7d356e30452249365fba99"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.1"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "ccd479984c7838684b3ac204b716c89955c76623"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "90630efff0894f8142308e334473eba54c433549"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.5.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "e27c4ebe80e8699540f2d6c805cc12203b614f12"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.20"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "87519eb762f85534445f5cda35be12e32759ee14"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.4"

[[deps.FiniteDiff]]
deps = ["ArrayInterfaceCore", "LinearAlgebra", "Requires", "Setfield", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "5a2cff9b6b77b33b89f3d97a4d367747adce647e"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.15.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "187198a4ed8ccd7b5d99c41b69c679269ea2b2d4"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.32"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.FunctionWrappers]]
git-tree-sha1 = "241552bc2209f0fa068b6415b1942cc0aa486bcc"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.2"

[[deps.FunctionWrappersWrappers]]
deps = ["FunctionWrappers"]
git-tree-sha1 = "a5e6e7f12607e90d71b09e6ce2c965e41b337968"
uuid = "77dc65aa-8811-40c2-897b-53d922fa7daf"
version = "0.1.1"

[[deps.Functors]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "a2657dd0f3e8a61dbe70fc7c122038bd33790af5"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.3.0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "6872f5ec8fd1a38880f027a26739d42dcda6691f"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.2"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "c98aea696662d09e215ef7cda5296024a9646c75"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.64.4"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "0eb5ef6f270fb70c2d83ee3593f56d02ed6fc7ff"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.68.0+0"

[[deps.GeoInterface]]
deps = ["Extents"]
git-tree-sha1 = "fb28b5dc239d0174d7297310ef7b84a11804dfab"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.0.1"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "12a584db96f1d460421d5fb8860822971cdb8455"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.4"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions", "Test"]
git-tree-sha1 = "709d864e3ed6e3545230601f94e11ebc65994641"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.11"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "d19f9edd8c34760dca2de2b503f969d8700ed288"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.1.4"

[[deps.InplaceOps]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "50b41d59e7164ab6fda65e71049fee9d890731ff"
uuid = "505f98c9-085e-5b2c-8e89-488be7bf1f34"
version = "0.3.0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "f67b55b6447d36733596aea445a9f119e83498b6"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.5"

[[deps.IntervalSets]]
deps = ["Dates", "Random", "Statistics"]
git-tree-sha1 = "076bb0da51a8c8d1229936a1af7bdfacd65037e1"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.2"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "b3364212fb5d870f724876ffcd34dd8ec6d98918"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.7"

[[deps.InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IterableTables]]
deps = ["DataValues", "IteratorInterfaceExtensions", "Requires", "TableTraits", "TableTraitsUtils"]
git-tree-sha1 = "70300b876b2cebde43ebc0df42bc8c94a144e1b4"
uuid = "1c8ee90f-4401-5389-894e-7a04a3dc0f4d"
version = "1.0.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "0f960b1404abb0b244c1ece579a0ec78d056a5d1"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.15"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "9816b296736292a80b9a3200eb7fbb57aaa3917a"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.5"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LRUCache]]
git-tree-sha1 = "d64a0aff6691612ab9fb0117b0995270871c5dfc"
uuid = "8ac3fa9e-de4c-5943-b1dc-09c6b5f20637"
version = "1.3.0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "ab9aa169d2160129beb241cb2750ca499b4e90e9"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.17"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LeftChildRightSiblingTrees]]
deps = ["AbstractTrees"]
git-tree-sha1 = "fb6803dafae4a5d62ea5cab204b1e657d9737e7f"
uuid = "1d6d02ad-be62-4b6b-8a6d-2f90e265016e"
version = "0.2.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtask]]
deps = ["FunctionWrappers", "LRUCache", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "dfa6c5f2d5a8918dd97c7f1a9ea0de68c2365426"
uuid = "6f1fad26-d15e-5dc8-ae53-837a1d7b8c9f"
version = "0.7.5"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "7bbea35cec17305fc70a0e5b4641477dc0789d9d"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.2.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogDensityProblems]]
deps = ["ArgCheck", "DocStringExtensions", "Random", "Requires", "UnPack"]
git-tree-sha1 = "408a29d70f8032b50b22155e6d7776715144b761"
uuid = "6fdf6af0-433a-55f7-b3ed-c6c6e0b8df7c"
version = "1.0.2"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "94d9c52ca447e23eac0c0f074effbcd38830deb5"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.18"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "5d4d2d9904227b8bd66386c1138cf4d5ffa826bf"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "0.4.9"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "dedbebe234e06e1ddad435f5c6f4b85cd8ce55f7"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "2.2.2"

[[deps.MCMCChains]]
deps = ["AbstractMCMC", "AxisArrays", "Compat", "Dates", "Distributions", "Formatting", "IteratorInterfaceExtensions", "KernelDensity", "LinearAlgebra", "MCMCDiagnosticTools", "MLJModelInterface", "NaturalSort", "OrderedCollections", "PrettyTables", "Random", "RecipesBase", "Serialization", "Statistics", "StatsBase", "StatsFuns", "TableTraits", "Tables"]
git-tree-sha1 = "0995883c615e93187e8365e35af771afcf74da03"
uuid = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
version = "5.4.0"

[[deps.MCMCDiagnosticTools]]
deps = ["AbstractFFTs", "DataAPI", "Distributions", "LinearAlgebra", "MLJModelInterface", "Random", "SpecialFunctions", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "59ac3cc5c08023f58b9cd6a5c447c4407cede6bc"
uuid = "be115224-59cd-429b-ad48-344e309966f0"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "41d162ae9c868218b1f3fe78cba878aa348c2d26"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.1.0+0"

[[deps.MLJModelInterface]]
deps = ["Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "16fa7c2e14aa5b3854bc77ab5f1dbe2cdc488903"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.6.0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "6872f9594ff273da6d13c7c1a1545d5a8c7d0c1c"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.6"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[deps.MicroCollections]]
deps = ["BangBang", "InitialValues", "Setfield"]
git-tree-sha1 = "6bb7786e4f24d44b4e29df03c69add1b63d88f01"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "50310f934e55e5ca3912fb941dec199b49ca9b68"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.2"

[[deps.NNlib]]
deps = ["Adapt", "ChainRulesCore", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "415108fd88d6f55cedf7ee940c7d4b01fad85421"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.8.9"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

[[deps.NamedArrays]]
deps = ["Combinatorics", "DataStructures", "DelimitedFiles", "InvertedIndices", "LinearAlgebra", "Random", "Requires", "SparseArrays", "Statistics"]
git-tree-sha1 = "2fd5787125d1a93fbe30961bd841707b8a80d75b"
uuid = "86f7a689-2022-50b4-a561-43c23ac3c673"
version = "0.9.6"

[[deps.NaturalSort]]
git-tree-sha1 = "eda490d06b9f7c00752ee81cfa451efe55521e21"
uuid = "c020b1a1-e9b0-503a-9c33-f039bfc54a85"
version = "1.0.0"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "1ea784113a6aa054c5ebd95945fa5e52c2f378e7"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.7"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e60321e3f2616584ff98f0a4f18d98ae6f89bbb3"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.17+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "ad8de074ed5dad13e87d76c467a82e5eff9c693a"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.7.2"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "1ef34738708e3f31994b52693286dabcb3d29f6b"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.2.9"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "cf494dca75a69712a72b80bc48f59dcf3dea63ec"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.16"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "3d5bf43e3e8b412656404ed9466f1dcbf7c50269"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.4.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "SnoopPrecompile", "Statistics"]
git-tree-sha1 = "21303256d239f6b484977314674aef4bb1fe4420"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.1"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "23d109aad5d225e945c813c6ebef79104beda955"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.26.0"

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "072cdf20c9b0507fdd977d7d246d90030609674b"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.0.5"

[[deps.PlutoLinks]]
deps = ["FileWatching", "InteractiveUtils", "Markdown", "PlutoHooks", "Revise", "UUIDs"]
git-tree-sha1 = "0e8bcc235ec8367a8e9648d48325ff00e4b0a545"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0420"
version = "0.1.5"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "LaTeXStrings", "Latexify", "Markdown", "PlutoLinks", "PlutoUI", "Random"]
git-tree-sha1 = "67c917d383c783aeadd25babad6625b834294b30"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.1.7"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "bf0a1121af131d9974241ba53f601211e9303a9e"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.37"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "d7a7aef8f8f2d537104f170139553b14dfe39fe9"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.2"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "c6c0f690d0cc7caddb74cef7aa847b824a16b256"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "3c009334f45dfd546a16a57960a821a1a023d241"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.5.0"

[[deps.Query]]
deps = ["DataValues", "IterableTables", "MacroTools", "QueryOperators", "Statistics"]
git-tree-sha1 = "a66aa7ca6f5c29f0e303ccef5c8bd55067df9bbe"
uuid = "1a8c2f83-1ff3-5112-b086-8aa67b057ba1"
version = "1.0.0"

[[deps.QueryOperators]]
deps = ["DataStructures", "DataValues", "IteratorInterfaceExtensions", "TableShowUtils"]
git-tree-sha1 = "911c64c204e7ecabfd1872eb93c49b4e7c701f02"
uuid = "2aef5ad7-51ca-5a8f-8e88-e75cf067b44b"
version = "0.9.3"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "dc84268fe0e3335a62e315a3a7cf2afa7178a734"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.3"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "dc1e451e15d90347a7decc4221842a022b011714"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.5.2"

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterfaceCore", "ArrayInterfaceStaticArraysCore", "ChainRulesCore", "DocStringExtensions", "FillArrays", "GPUArraysCore", "IteratorInterfaceExtensions", "LinearAlgebra", "RecipesBase", "StaticArraysCore", "Statistics", "Tables", "ZygoteRules"]
git-tree-sha1 = "3004608dc42101a944e44c1c68b599fa7c669080"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "2.32.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "cdbd3b1338c72ce29d9584fdbe9e9b70eeb5adca"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.1.3"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Revise]]
deps = ["CodeTracking", "Distributed", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Pkg", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "dad726963ecea2d8a81e26286f625aee09a91b7c"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.4.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.Roots]]
deps = ["CommonSolve", "Printf", "Setfield"]
git-tree-sha1 = "b3fb8294be9d311c9b3fa8df2f1f31c93d40340a"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "2.0.4"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.SciMLBase]]
deps = ["ArrayInterfaceCore", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "FunctionWrappersWrappers", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "Preferences", "RecipesBase", "RecursiveArrayTools", "StaticArraysCore", "Statistics", "Tables"]
git-tree-sha1 = "e6778c4d41f3d6213bf4d2803c4eb9ef12b6c0a7"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "1.59.3"

[[deps.ScientificTypesBase]]
git-tree-sha1 = "a8e18eb383b5ecf1b5e6fc237eb39255044fd92b"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "3.0.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "f94f779c94e58bf9ea243e77a37e16d9de9126bd"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "130c68b3497094753bacf084ae59c9eeaefa2ee7"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.14"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "38d88503f695eb0301479bc9b0d4320b378bafe5"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "0.8.2"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SnoopPrecompile]]
git-tree-sha1 = "f604441450a3c0569830946e5b33b78c928e1a85"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.1"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "39c9f91521de844bad65049efd4f9223e7ed43f9"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.14"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "2189eb2c1f25cb3f43e5807f26aa864052e50c17"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.8"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "30b9236691858e13f167ce829490a68e1a597782"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "3.2.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5783b877201a82fc0014cbf381e7e6eb130473a4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.0.1"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArraysCore", "Tables"]
git-tree-sha1 = "8c6ac65ec9ab781af05b08ff305ddc727c25f680"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.12"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableShowUtils]]
deps = ["DataValues", "Dates", "JSON", "Markdown", "Test"]
git-tree-sha1 = "14c54e1e96431fb87f0d2f5983f090f1b9d06457"
uuid = "5e66a065-1f0a-5976-b372-e0b8c017ca10"
version = "0.2.5"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.TableTraitsUtils]]
deps = ["DataValues", "IteratorInterfaceExtensions", "Missings", "TableTraits"]
git-tree-sha1 = "78fecfe140d7abb480b53a44f3f85b6aa373c293"
uuid = "382cd787-c1b6-5bf2-a167-d5b971a19bda"
version = "1.0.2"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "7149a60b01bf58787a1b83dad93f90d4b9afbe5d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.8.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.TerminalLoggers]]
deps = ["LeftChildRightSiblingTrees", "Logging", "Markdown", "Printf", "ProgressLogging", "UUIDs"]
git-tree-sha1 = "f53e34e784ae771eb9ccde4d72e578aa453d0554"
uuid = "5d786b92-1e48-4d6f-9151-6b4477ca9bed"
version = "0.1.6"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Tracker]]
deps = ["Adapt", "DiffRules", "ForwardDiff", "Functors", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NNlib", "NaNMath", "Optimisers", "Printf", "Random", "Requires", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "d963aad627fd7af56fbbfee67703c2f7bfee9dd7"
uuid = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
version = "0.2.22"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "8a75929dcd3c38611db2f8d08546decb514fcadf"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.9"

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "c76399a3bbe6f5a88faa33c8f8a65aa631d95013"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.73"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.Turing]]
deps = ["AbstractMCMC", "AdvancedHMC", "AdvancedMH", "AdvancedPS", "AdvancedVI", "BangBang", "Bijectors", "DataStructures", "Distributions", "DistributionsAD", "DocStringExtensions", "DynamicPPL", "EllipticalSliceSampling", "ForwardDiff", "Libtask", "LinearAlgebra", "LogDensityProblems", "MCMCChains", "NamedArrays", "Printf", "Random", "Reexport", "Requires", "SciMLBase", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Tracker"]
git-tree-sha1 = "68fb67dab0c11de2bb1d761d7a742b965a9bc875"
uuid = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"
version = "0.21.12"

[[deps.URIs]]
git-tree-sha1 = "e59ecc5a41b000fa94423a578d29290c7266fc10"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "58443b63fb7e465a8a7210828c91c08b92132dff"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.14+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9ebfc140cc56e8c2156a15ceac2f0302e327ac0a"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+0"
"""

# ╔═╡ Cell order:
# ╟─82d5eb4f-5724-4c72-b6e0-f6d5fc7f4313
# ╟─57141374-dd5a-4eaa-8235-b2310ef2d600
# ╟─921f13df-bc87-4d1f-8429-90cd234a65a1
# ╟─4ce8dd30-9b3e-4411-9a43-dcd77149aea2
# ╟─6c01ab20-217f-4671-9ade-8ac928a65771
# ╟─6e34f8e1-99cb-4557-9537-2e33ee864267
# ╟─4bdcca25-c37f-4079-b222-be773adc2b8f
# ╟─ee7aaab9-5e4f-46ab-8100-75be142fba72
# ╟─2306a2d5-2924-45e0-adec-b90d536d2949
# ╟─bab9033c-b9ee-45c1-9466-838e40bdb920
# ╟─3d1821a6-f134-49d6-a4b0-39d6d28ab420
# ╟─9945831b-96ca-4eb4-8993-4acb0dc4b08e
# ╟─2e51744b-b040-4f21-94b8-ffe9cd1e149e
# ╟─d79fc353-e30e-49ab-aa8e-9ba4b76a879b
# ╠═9404128d-0638-45ba-aaf6-a6ea47489b49
# ╠═253decc1-35c7-4454-b500-4f28e1087d36
# ╠═5e92054a-ca9e-4949-9727-5a9ed14003c0
# ╟─ffd80564-ea2a-40c3-8250-5c9482ab641d
# ╠═bce3f35c-07a1-48ef-8a29-243b2215fcb5
# ╟─174f6c6d-6ff9-449d-86b3-85bcad9f01a2
# ╟─8b1f8b91-12b5-4e61-a8ff-63538189cf34
# ╟─5edc2a2d-6f63-4ac6-8c33-2c5d670bc466
# ╟─811ed6ac-4cf2-435a-934c-edfbb38564b2
# ╟─b53fc91a-d2ed-4727-a683-205092e33bc6
# ╟─10f95d69-9cd8-47d4-a534-8de09ea3b216
# ╟─b821a2ae-bf16-4018-b85a-ff1713f40103
# ╠═2f09622b-838c-42df-a74f-81960916fae2
# ╟─21834080-14de-4926-9766-5a3ad994e2a1
# ╟─3c14ec5c-e72a-4af4-8859-fd7a0bf91409
# ╠═fcf19e04-3e35-4a01-8036-fd5b283fdd37
# ╟─5d61ebb8-465a-4d10-a0e6-a0c043f511b5
# ╠═44eb9ddd-e04f-4961-b11e-50df17731516
# ╟─99e98b44-1994-4d0f-ba38-f10887a1be0c
# ╠═eb83fe44-d67d-43cd-b823-b29b965a17af
# ╟─49fdca20-46fd-4f31-94f1-ed58f3b32305
# ╠═41b2eea0-3049-4fa5-803e-83a54b74ef27
# ╟─29352332-7fae-4709-9883-cfb480650a6c
# ╠═82e06e8b-8211-48bc-a719-743a616a2c8b
# ╟─43ae8d15-6381-4c86-b08d-2d12cd4bc653
# ╟─0a508687-c2b7-466d-a5d6-2d1792687f3a
# ╟─abc38d23-8665-4377-9a25-9e9c5a10a7bf
# ╟─844ede38-9596-47a6-b30b-9eff622a2330
# ╟─cf582fc9-07b1-4d09-b379-2576924c026b
# ╠═9007735e-45c2-4a96-8ea5-03d7b8b58410
# ╟─ba869a69-167e-4a1c-92af-e8592f6fca3d
# ╠═220caa90-90e8-4a52-a133-e37bb9cf5b50
# ╟─11469768-34af-470a-b431-c47b17d6a586
# ╟─710af3aa-b842-43b2-ab96-cda80b2a2ee0
# ╟─b89645c8-6574-4f53-b40e-5c4e4236671e
# ╟─b9d0ecc6-f9a8-4107-9945-f17aa09e0b87
# ╟─ebec340e-297f-44c1-8095-60ea68dd530c
# ╟─bed8ac6f-052a-4cb3-9fb8-5162f1683dd2
# ╟─a9a0bf9f-4ab5-42c5-aa5e-24678ba5ca5a
# ╟─7296def6-31c0-4df2-b6ca-2fd953bdfb1f
# ╟─4c890552-529c-46e1-bb1e-cbcea7f24672
# ╟─d305c716-080f-4faa-9143-bc75bfa6ce6f
# ╠═0737daef-b8f1-49ef-9a06-5cf1b716f719
# ╟─10a2b5b5-6a84-4b1e-a5f6-dd2434541edb
# ╠═390d9fc3-22f1-4e46-8164-4fc33f494035
# ╟─8743b110-ed40-4718-8fc3-e296ee8339f2
# ╟─67f3aef3-f34f-4e67-8ff4-adb8aa0284db
# ╟─6a141962-d4d6-4f27-b94e-2d0aee0740c7
# ╟─5c9f6b52-87d3-4971-90f4-3f953f7bce7f
# ╠═5d631862-97e7-4ccd-ab6f-a875989dde99
# ╠═1e63307b-bb3b-4b32-9983-90626e1a3dde
# ╟─1af9fbff-4786-415c-b1e6-ecc580e22ef2
# ╟─fc8d934c-7557-4011-a1e1-961bfd4e11a1
# ╠═9371ce38-9cb5-4664-ae24-5554f4847868
# ╟─bd82489d-5156-4d88-98ea-8f05eab22e00
# ╟─86aa15f3-b525-4c77-ab01-159b8e977314
# ╟─962d20ef-8446-4894-80df-725c1bac04be
# ╟─6c2ab644-4799-4bf3-aac0-431bf399dab0
# ╟─300a1ea8-023e-4c38-8a7e-1bc9ac62c311
# ╟─e2905cac-57f9-4da9-9160-828f46d72bb6
# ╟─7e1a46fd-392c-412c-8a5c-e54765112564
# ╠═a935174a-1057-4ad6-9b92-84981f4a4bb2
# ╟─eb96986f-78fe-4a28-9ccf-6d3a66f063a6
# ╟─3cf57331-688b-4b71-83f1-51cf53cfb0ee
# ╟─7a2e757b-4117-455d-ba41-6205ec4746dd
# ╟─60a09c22-4100-41e9-8ecc-bfefde1492e0
# ╟─c7fe279b-d978-4b97-a031-bb933b64d90f
# ╟─121869f2-c78d-4b46-bd5d-9d97a2f68e54
# ╠═96fc2d52-5128-483c-9962-817f1b013065
# ╟─c7d3155c-5124-4724-bc7a-f1e2dcde879b
# ╟─abad7a4d-0bb8-4c8f-bdec-f9e0d2839fd7
# ╠═afadd762-5eb8-47ca-82b3-0862299e5fb9
# ╟─e325a28f-c8ef-4f0c-8f29-e9f4c34ea746
# ╠═244eccc2-463d-453b-bc30-1decbf0eed9a
# ╟─bb1e1664-0c67-4aea-9e76-37669d253592
# ╠═66f4acf5-152c-4792-9d7f-9a0ddb6459f6
# ╟─b60aadbc-4e70-414e-9fdc-c3b042cb17bf
# ╠═8be9bf52-a0a3-11ec-045f-3962ad227049
# ╟─69f40924-6b24-4014-8c1b-f600a0759aab
# ╟─5f438b58-2f87-4373-b9ca-e35673b7b46f
# ╟─cc7006c7-e3ef-470a-b93e-5743a27a32d9
# ╟─3fbcc50d-9f6a-4aec-9a8f-f2f525223f0e
# ╟─a7514405-af4c-4f16-8508-91ee624d8a1c
# ╟─690205fb-0b95-4614-9b66-dec362ed693c
# ╟─4f047081-a4d6-414b-9c3e-0eb055c730b3
# ╟─8f700e72-df0f-4e68-85fe-7fbe8da7fbb1
# ╟─7047d464-efdd-4315-b930-5b2e8a3d93c5
# ╟─e33e5fdc-7f14-4469-b462-6766ad3ce230
# ╟─56d09fea-e2c2-4345-a089-419ac863ac43
# ╟─3932fb9d-2897-4d64-8dba-a51799d1aa7a
# ╟─d4e5cd92-21c0-4073-ab3e-8cd5804976c8
# ╟─71a560d1-efb4-48ad-8a45-ac4fd64537fd
# ╟─9283ac8f-c45e-42e0-ac41-579a3dd35e96
# ╟─14cca39b-cded-4108-877f-842c70ff8843
# ╟─50bdc155-4e41-4f8f-8b6e-12599138ec80
# ╟─65c6dea3-abe1-4226-a943-02d5a3138873
# ╟─91d39e07-5507-41e2-89b4-4d5236f70cf8
# ╟─fd0f40ee-1e1a-4a04-aa89-dbe30cd387b3
# ╟─fcb2cae4-5f57-4c77-a463-3112a2853280
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002

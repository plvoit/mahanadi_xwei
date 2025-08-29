# Paul Voit 4 October 2024
# This script fits the dGEV to each cell in a grid
# The R and scipy fits use the oppsite sign for the
# shape parameter. this also has to be fixed before using the Rfits in the
# python workflow


library(ncdf4)
library(IDF)
library(hash)

path = "/path/to/your/workdir"

# Get the command-line arguments passed to the script
args <- commandArgs(trailingOnly = TRUE)
# The variable is the first argument
location <- as.character(args[1])

setwd(path)

#for the results
dir.create("output/dgev_original")
dir.create(paste0("output/dgev_original/", location))

files = sort(dir(paste0(path, "output/yearmax/", location), pattern='*.nc', full.names = TRUE))

#logfile
cat("Error log for dGEV fit ", file="error.log", append=FALSE, sep = "\n")

# these are netcdf where I already got the yearly maxima of the rainfall for respective durations.
#the yearmax need to be divided by duration to get the intensities

# xarray sorts the dimensions [time, lat, lon]
# R sorts the dimensions [lon, lat, time]
y_01 = nc_open(files[1])
rain_dims <- y_01$dim

# Assuming the dimensions are in the order (time, lat, lon),
# get the dimensions of lat and lon and reshape accordingly.
lat_dim <- rain_dims[["latitude"]]$len
lon_dim <- rain_dims[["longitude"]]$len

#@Saikat: Here we were using durations from 1-90 days for the rainfall accumulation
# Has to be changed accordingly
y_01 = ncvar_get(y_01, "Rainfall")
# Reshape the array back to its original dimensions
#this has to be done to prevent a dimension of length = 1 from disappearing
y_01 <- array(y_01, dim = sapply(rain_dims, function(x) x$len))
#we're doing this to keep the same order that xarray would use when making an array out of a netcdf (y,x)
y_01 <- aperm(y_01, c(2, 1, 3))

y_03 = nc_open(files[2])
y_03 = ncvar_get(y_03, "Rainfall") / 3
y_03 <- array(y_03, dim = sapply(rain_dims, function(x) x$len))
y_03 <- aperm(y_03, c(2, 1, 3))

y_06 = nc_open(files[3])
y_06 = ncvar_get(y_06, "Rainfall") / 6
y_06 <- array(y_06, dim = sapply(rain_dims, function(x) x$len))
y_06 <- aperm(y_06, c(2, 1, 3))

y_30 = nc_open(files[4])
y_30 = ncvar_get(y_30, "Rainfall") / 30
y_30 <- array(y_30, dim = sapply(rain_dims, function(x) x$len))
y_30 <- aperm(y_30, c(2, 1, 3))

y_60 = nc_open(files[5])
y_60 = ncvar_get(y_60, "Rainfall") / 60
y_60 <- array(y_60, dim = sapply(rain_dims, function(x) x$len))
y_60 <- aperm(y_60, c(2, 1, 3))

y_90 = nc_open(files[6])
y_90 = ncvar_get(y_90, "Rainfall") / 90
y_90 <- array(y_90, dim = sapply(rain_dims, function(x) x$len))
y_90 <- aperm(y_90, c(2, 1, 3))


h <- hash()

h[["01"]] <- y_01
h[["03"]] <- y_03
h[["06"]] <- y_06
h[["30"]] <- y_30
h[["60"]] <- y_60
h[["90"]] <- y_90



##Total number of years in the yearmaxima timeseries
#to get the length of the dimensions right, explicitly call them by name
ncfile = nc_open(files[1])
nr_years = ncfile$dim$time$len
len_lat = ncfile$dim$latitude$len
len_lon = ncfile$dim$longitude$len

skip_error <- function(i, j){
  cat(paste0("Error in i: ", i, " j: ", j), file="error.log", append=TRUE, sep = "\n")
  print(paste0("Error in i: ", i, " j: ", j))
  return(-999)
  }


dgev_fit <- function(data_chunk){
  rows = len_lat
  cols = len_lon
  
  mod_loc = array(NA, dim=c(rows, cols))
  scale_0 = array(NA, dim=c(rows, cols))
  shape = array(NA, dim=c(rows, cols))
  duration_offset = array(NA, dim=c(rows, cols))
  duration_exp = array(NA, dim=c(rows, cols))
  
  #the order is so confusing in R. Here its lon, lat, time. Since we want the resulting
  # array to be lat (rows), lon(columnsy)
  for(row in 1:rows){
    for(col in 1:cols){

    #@Saikat: This NA rule has to be changed because your yearmax series is much longer.
    # Our data was 30 years or something like that...
    #There can be no Nas in xdat. We say: at least 15 values should be there for a duration
    #count NAs for each duration so we can create an according ds vector
    #This was changed Aug 2024. Previously the following statement was used:
    #if (any(is.na(xdat))) next

      #Take out the NAs
      x_dat_01 <- data_chunk[["01"]][row, col,][!is.na(data_chunk[["01"]][row, col,])]
      x_dat_03 <- data_chunk[["03"]][row, col,][!is.na(data_chunk[["03"]][row, col,])]
      x_dat_06 <- data_chunk[["06"]][row, col,][!is.na(data_chunk[["06"]][row, col,])]
      x_dat_30 <- data_chunk[["30"]][row, col,][!is.na(data_chunk[["30"]][row, col,])]
      x_dat_60 <- data_chunk[["60"]][row, col,][!is.na(data_chunk[["60"]][row, col,])]
      x_dat_90 <- data_chunk[["90"]][row, col,][!is.na(data_chunk[["90"]][row, col,])]

      #check if any duration has less than 15 values. In this case we kick the 
      #cell out
      l <- c(length(x_dat_01), length(x_dat_03), length(x_dat_06),
            length(x_dat_30), length(x_dat_60),length(x_dat_90))
      
      if(any(l <= 10)) next
      
      #build xdat and according ds
      xdat <- c(x_dat_01,
                x_dat_03,
                x_dat_06,
                x_dat_30,
                x_dat_60,
                x_dat_90)

      ds <- c(rep(1,l[[1]]), rep(3,l[[2]]), rep(6,l[[3]]), rep(30,l[[4]]), rep(60,l[[5]]),
              rep(90,l[[6]]))
      
      
      fit <-  IDF::gev.d.fit(xdat, ds, sigma0link = make.link('log'), show=FALSE)
      
      #somehow some fits throw an error. We log them and continue
      tryCatch(
    
        expr = {
          fit <-  gev.d.fit(xdat,ds, sigma0link = make.link('log'), show=FALSE)},
       
        error = function(e){
          fit <<- skip_error(row, col)}
      )

      if(is.numeric(fit)){
        if(fit == -999) next}
 
      params <- IDF::gev.d.params(fit)
    
      mod_loc[row, col] <- params$mut
      scale_0[row, col] <- params$sigma0
      shape[row, col] <- params$xi
      duration_offset[row, col] <- params$theta
      duration_exp[row, col] <- params$eta
    }
  }
  
  return(list(mod_loc, scale_0, shape, duration_offset, duration_exp))
}


# Fit GEV to all cell
result_list <- dgev_fit(h)

#put the results back together
res_mod_loc <- as.matrix(result_list[[1]])
res_scale_0 <- as.matrix(result_list[[2]])
res_shape <- as.matrix(result_list[[3]]) * -1 #needs to be negated to work with scipy
res_duration_offset <- as.matrix(result_list[[4]])
res_duration_exp <- as.matrix(result_list[[5]])


# these arrays are rotated by 90°
write.table(res_scale_0, paste0("output/dgev_original/", location,'/scale_0.csv'), col.names = F, row.names = F, sep = ",")
write.table(res_mod_loc,paste0("output/dgev_original/", location, '/mod_loc.csv'), col.names = F, row.names = F, sep = ",")
write.table(res_shape, paste0("output/dgev_original/", location, '/shape.csv'), col.names = F, row.names = F, sep = ",")
write.table(res_duration_offset,paste0("output/dgev_original/", location, '/duration_offset.csv'), col.names = F, row.names = F, sep = ",")
write.table(res_duration_exp, paste0("output/dgev_original/", location, '/duration_exp.csv'), col.names = F, row.names = F, sep = ",")

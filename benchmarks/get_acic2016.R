# ACIC 2016 (Dorie et al. 2019) with the official aciccomp2016 generator.
# Writes data/ACIC2016/x.csv (the competition covariates) and zymu_<i>.csv for
# the 77 parameter settings, one replication each (random.seed = 1), with
# columns z, y0, y1, mu0, mu1. The paper uses settings 1-40.
suppressMessages(library(aciccomp2016))
out <- file.path("data", "ACIC2016")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
stopifnot(identical(dim(input_2016), c(4802L, 58L)))
write.csv(input_2016, file.path(out, "x.csv"), row.names = FALSE)
for (i in seq_len(nrow(parameters_2016))) {
  d <- dgp_2016(input_2016, i, random.seed = 1L)
  write.csv(data.frame(z = d$z, y0 = d$y.0, y1 = d$y.1, mu0 = d$mu.0, mu1 = d$mu.1),
            file.path(out, sprintf("zymu_%d.csv", i)), row.names = FALSE)
}
cat(sprintf("wrote x.csv and %d settings to %s\n", nrow(parameters_2016), out))

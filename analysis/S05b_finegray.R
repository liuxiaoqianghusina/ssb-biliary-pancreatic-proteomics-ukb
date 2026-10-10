# Fine-Gray subdistribution hazard models with death as a competing event (survival::finegray + weighted coxph).
# Inputs from S05_extra.py: cache/finegray_intake.csv (SSB + ASB per serving, M2 covariates z_*) and
# cache/finegray_signature.csv (signatures per SD, M2 covariates). Status: 0 censored, 1 event, 2 death without event.
suppressPackageStartupMessages({library(survival); library(data.table)})
P <- paste0(Sys.getenv("SSB_PROJECT_DIR", "."), "/")
outs <- c("Gallstones", "AP", "AP_biliary", "AP_nonbiliary", "CP")
res <- list()
run <- function(file, xs, label, together) {
  d <- fread(paste0(P, "cache/", file))
  z <- grep("^z_", names(d), value = TRUE)
  z <- z[sapply(z, function(v) sd(d[[v]], na.rm = TRUE) > 0)]
  sets <- if (together) list(xs) else as.list(xs)
  for (oc in outs) {
    dd <- d[get(paste0("elig_", oc)) == 1]
    dd[, status := factor(get(paste0("S_", oc)), levels = 0:2, labels = c("censor", "event", "death"))]
    dd[, time := get(paste0("T_", oc))]
    zz <- z[sapply(z, function(v) sd(dd[[v]]) > 0)]
    for (x in sets) {
      f <- as.formula(paste("Surv(time, status) ~", paste(c(x, zz), collapse = " + ")))
      fg <- finegray(f, data = as.data.frame(dd), etype = "event")
      m <- coxph(as.formula(paste("Surv(fgstart, fgstop, fgstatus) ~", paste(c(x, zz), collapse = " + "))),
                 data = fg, weights = fgwt)
      for (v in x) {
        co <- summary(m)$coefficients[v, ]; ci <- summary(m)$conf.int[v, ]
        res[[length(res) + 1]] <<- data.table(analysis = label, outcome = oc, exposure = v, n = nrow(dd),
                                              events = sum(dd$status == "event"), deaths = sum(dd$status == "death"),
                                              shr = ci["exp(coef)"], lci = ci["lower .95"], uci = ci["upper .95"],
                                              p = co["Pr(>|z|)"])
        cat(label, oc, v, sprintf("%.2f (%.2f-%.2f)", ci["exp(coef)"], ci["lower .95"], ci["upper .95"]), "\n")
      }
    }
  }
}
run("finegray_signature.csv", c("sig_SSB", "sig_ASB"), "signature per SD (M2)", FALSE)
run("finegray_intake.csv", c("SSB", "ASB"), "intake per serving (M2)", TRUE)
fwrite(rbindlist(res), paste0(P, "results/finegray.csv"))

"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Progress } from "@/components/ui/progress"

export default function ExoplanetConfidencePredictor() {
  const [inputs, setInputs] = useState({
    sy_pnum: "",
    pl_orbper: "",
    pl_orbsmax: "",
    pl_rade: "",
    pl_radj: "",
    pl_masse: "",
    pl_massj: "",
    pl_dens: "",
    pl_eqt: "",
    pl_orbincl: "",
    st_teff: "",
    st_rad: "",
    st_mass: "",
    st_met: "",
    st_lum: "",
    sy_dist: "",
    sy_vmag: "",
    sy_gmag: "",
  })

  const [confidence, setConfidence] = useState<number | null>(null)
  const [isCalculating, setIsCalculating] = useState(false)

  const parameters = [
    { key: "sy_pnum", label: "Number of Planets in System", unit: "count", category: "System" },
    { key: "pl_orbper", label: "Orbital Period", unit: "days", category: "Planet" },
    { key: "pl_orbsmax", label: "Orbital Semi-Major Axis", unit: "AU", category: "Planet" },
    { key: "pl_rade", label: "Planet Radius (Earth Radii)", unit: "R⊕", category: "Planet" },
    { key: "pl_radj", label: "Planet Radius (Jupiter Radii)", unit: "R♃", category: "Planet" },
    { key: "pl_masse", label: "Planet Mass (Earth Masses)", unit: "M⊕", category: "Planet" },
    { key: "pl_massj", label: "Planet Mass (Jupiter Masses)", unit: "M♃", category: "Planet" },
    { key: "pl_dens", label: "Planet Density", unit: "g/cm³", category: "Planet" },
    { key: "pl_eqt", label: "Equilibrium Temperature", unit: "K", category: "Planet" },
    { key: "pl_orbincl", label: "Orbital Inclination", unit: "degrees", category: "Planet" },
    { key: "st_teff", label: "Stellar Effective Temperature", unit: "K", category: "Star" },
    { key: "st_rad", label: "Stellar Radius", unit: "R☉", category: "Star" },
    { key: "st_mass", label: "Stellar Mass", unit: "M☉", category: "Star" },
    { key: "st_met", label: "Stellar Metallicity", unit: "[Fe/H]", category: "Star" },
    { key: "st_lum", label: "Stellar Luminosity", unit: "L☉", category: "Star" },
    { key: "sy_dist", label: "System Distance", unit: "pc", category: "System" },
    { key: "sy_vmag", label: "V-band Magnitude", unit: "mag", category: "System" },
    { key: "sy_gmag", label: "G-band Magnitude", unit: "mag", category: "System" },
  ]

  const handleInputChange = (key: string, value: string) => {
    setInputs((prev) => ({ ...prev, [key]: value }))
  }

  const calculateConfidence = async () => {
    setIsCalculating(true)
    await new Promise((resolve) => setTimeout(resolve, 1500))

    // ========================================
    // PASTE YOUR ALGORITHM HERE
    // ========================================
    //
    // The 'inputs' object contains all the parameter values:
    // - inputs.sy_pnum (number of planets)
    // - inputs.pl_orbper (orbital period)
    // - inputs.pl_orbsmax (orbital semi-major axis)
    // - inputs.pl_rade (planet radius in Earth radii)
    // - inputs.pl_radj (planet radius in Jupiter radii)
    // - inputs.pl_masse (planet mass in Earth masses)
    // - inputs.pl_massj (planet mass in Jupiter masses)
    // - inputs.pl_dens (planet density)
    // - inputs.pl_eqt (equilibrium temperature)
    // - inputs.pl_orbincl (orbital inclination)
    // - inputs.st_teff (stellar effective temperature)
    // - inputs.st_rad (stellar radius)
    // - inputs.st_mass (stellar mass)
    // - inputs.st_met (stellar metallicity)
    // - inputs.st_lum (stellar luminosity)
    // - inputs.sy_dist (system distance)
    // - inputs.sy_vmag (V-band magnitude)
    // - inputs.sy_gmag (G-band magnitude)
    //
    // Replace the code below with your algorithm
    // Your algorithm should return a confidence value between 0-100

    // TEMPORARY PLACEHOLDER - REPLACE THIS WITH YOUR ALGORITHM
    const filledInputs = Object.values(inputs).filter((val) => val !== "").length
    const completeness = filledInputs / parameters.length
    const baseConfidence = Math.random() * 40 + 30
    const adjustedConfidence = Math.min(95, baseConfidence + completeness * 25)
    const calculatedConfidence = Math.round(adjustedConfidence)

    // ========================================
    // END OF ALGORITHM SECTION
    // ========================================

    setConfidence(calculatedConfidence)
    setIsCalculating(false)
  }

  const clearInputs = () => {
    setInputs(Object.keys(inputs).reduce((acc, key) => ({ ...acc, [key]: "" }), {}))
    setConfidence(null)
  }

  const getConfidenceColor = (conf: number) => {
    if (conf >= 80) return "text-green-600"
    if (conf >= 60) return "text-yellow-600"
    return "text-red-600"
  }

  const getConfidenceLabel = (conf: number) => {
    if (conf >= 80) return "High Confidence"
    if (conf >= 60) return "Medium Confidence"
    return "Low Confidence"
  }

  // Group parameters by category
  const groupedParams = parameters.reduce(
    (acc, param) => {
      if (!acc[param.category]) acc[param.category] = []
      acc[param.category].push(param)
      return acc
    },
    {} as Record<string, typeof parameters>,
  )

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8 max-w-6xl">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-4 text-foreground">Exoplanet Habitability Predictor</h1>
          <p className="text-lg text-muted-foreground max-w-3xl mx-auto">
            Machine learning algorithm for predicting exoplanet habitability confidence based on astronomical
            parameters.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          <div className="lg:col-span-3 space-y-6">
            {Object.entries(groupedParams).map(([category, params]) => (
              <Card key={category} className="border border-border">
                <CardHeader>
                  <CardTitle className="text-lg font-semibold">{category} Parameters</CardTitle>
                  <CardDescription>Enter {category.toLowerCase()} characteristics</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {params.map((param) => (
                      <div key={param.key} className="space-y-2">
                        <Label htmlFor={param.key} className="text-sm font-medium">
                          {param.label}
                          <span className="text-muted-foreground font-normal ml-1">({param.unit})</span>
                        </Label>
                        <Input
                          id={param.key}
                          type="number"
                          step="any"
                          placeholder={`Enter ${param.label.toLowerCase()}`}
                          value={inputs[param.key as keyof typeof inputs]}
                          onChange={(e) => handleInputChange(param.key, e.target.value)}
                          className="h-10 border-border focus:border-ring focus:ring-ring"
                        />
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            ))}

            <div className="flex gap-4">
              <Button
                onClick={calculateConfidence}
                disabled={isCalculating}
                className="flex-1 h-12 bg-primary text-primary-foreground hover:bg-primary/90"
              >
                {isCalculating ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                    Calculating...
                  </>
                ) : (
                  "Calculate Confidence"
                )}
              </Button>
              <Button
                variant="outline"
                onClick={clearInputs}
                className="h-12 border-border hover:bg-accent hover:text-accent-foreground bg-transparent"
              >
                Clear All
              </Button>
            </div>
          </div>

          <div className="space-y-6">
            <Card className="border border-border">
              <CardHeader>
                <CardTitle className="text-lg font-semibold">Results</CardTitle>
                <CardDescription>Confidence prediction</CardDescription>
              </CardHeader>
              <CardContent>
                {confidence !== null ? (
                  <div className="space-y-4">
                    <div className="text-center">
                      <div className={`text-4xl font-bold mb-2 ${getConfidenceColor(confidence)}`}>{confidence}%</div>
                      <div className="text-sm text-muted-foreground bg-muted px-3 py-1 inline-block">
                        {getConfidenceLabel(confidence)}
                      </div>
                    </div>
                    <Progress value={confidence} className="h-3" />
                    <div className="text-center text-sm text-muted-foreground">
                      Based on {Object.values(inputs).filter((val) => val !== "").length} of {parameters.length}{" "}
                      parameters
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <div className="text-muted-foreground mb-2">No prediction yet</div>
                    <p className="text-sm text-muted-foreground">Fill in parameters above and click Calculate</p>
                  </div>
                )}
              </CardContent>
            </Card>

            <Card className="border border-border">
              <CardHeader>
                <CardTitle className="text-lg font-semibold">About</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4 text-sm text-muted-foreground">
                <div>
                  <h4 className="font-medium text-foreground mb-1">Dataset</h4>
                  <p>Exoplanet archive with confirmed planetary discoveries</p>
                </div>
                <div>
                  <h4 className="font-medium text-foreground mb-1">Algorithm</h4>
                  <p>Machine learning model for confidence prediction</p>
                </div>
                <div>
                  <h4 className="font-medium text-foreground mb-1">Parameters</h4>
                  <p>{parameters.length} features covering orbital, physical, and observational data</p>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}
